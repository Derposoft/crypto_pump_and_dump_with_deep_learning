import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anomaly_transformer_embed import DataEmbedding, TokenEmbedding


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list

# implementation from https://github.com/thuml/Anomaly-Transformer
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, series_prior_loss_weight=1.0,):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.series_prior_loss_weight = series_prior_loss_weight
        self.win_size = win_size

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out  # [B, L, D]

    # This is different than original paper
    def loss_fn(self, x, y):
        out, series, prior, _ = x
        
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += (torch.mean(my_kl_loss(series[u], (
                    prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)).detach())) + torch.mean(
                my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            self.win_size)).detach(),
                    series[u])))
            prior_loss += (torch.mean(
                my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)),
                            series[u].detach())) + torch.mean(
                my_kl_loss(series[u].detach(),
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    self.win_size)))))
        series_loss = series_loss / len(prior)
        prior_loss = prior_loss / len(prior)

        return F.mse_loss(out, y) + (series_loss + prior_loss) * self.series_prior_loss_weight


# Old anomaly transformer implementations
class AnomalyTransformerBlock(nn.Module):
    def __init__(self, N, d_model, device):
        super().__init__()
        self.N, self.d_model = N, d_model

        self.attention = AnomalyAttention(self.N, self.d_model, device=device)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.ReLU())
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x_identity = x
        x = self.attention(x)
        z = self.ln1(x + x_identity)

        z_identity = z
        z = self.ff(z)
        z = self.ln2(z + z_identity)

        return z

class AnomalyTransfomerBasic(nn.Module):
    def __init__(self, N, d_model, layers, device, output_size=1):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, device) for _ in range(layers)]
        )
        self.classifier = nn.Linear(d_model, output_size)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x.squeeze(2)

class AnomalyTransfomerIntermediate(nn.Module):
    def __init__(self, N, d_model, layers, lambda_, device):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, device) for _ in range(layers)]
        )
        self.output = None
        self.lambda_ = lambda_
        self.classifier = nn.Linear(d_model, 1)

        self.P_layers = []
        self.S_layers = []

        self.input_ = None

    def forward(self, x):
        self.input_ = x
        self.P_layers = []
        self.S_layers = []
        for block in self.blocks:
            x = block(x)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)
        x = self.classifier(x)
        return x.squeeze(2)

    def layer_association_discrepancy(self, Pl, Sl):
        rowwise_kl = lambda row: (
            F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
        )
        ad_vector = torch.concat(
            [rowwise_kl(row).unsqueeze(0) for row in range(Pl.shape[0])]
        )
        return ad_vector

    def association_discrepancy(self, P_list, S_list):

        return (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy(P, S)
                for P, S in zip(P_list, S_list)
            ]
        )

    def loss_fn(self, preds, y):
        S_list = self.S_layers
        P_list = [P.detach() for P in self.P_layers]
        return F.mse_loss(preds, y) + self.lambda_*torch.mean(torch.abs(self.association_discrepancy(P_list, S_list)))

class AnomalyTransformerOld(nn.Module):
    def __init__(self, N, d_model, layers, lambda_, device):
        super().__init__()
        self.N = N
        self.d_model = d_model

        self.blocks = nn.ModuleList(
            [AnomalyTransformerBlock(self.N, self.d_model, device) for _ in range(layers)]
        )
        self.output = None
        self.lambda_ = lambda_
        self.classifier = nn.Linear(d_model, 1)

        self.P_layers = []
        self.S_layers = []

    def forward(self, x):
        self.P_layers = []
        self.S_layers = []
        for block in self.blocks:
            x = block(x)
            self.P_layers.append(block.attention.P)
            self.S_layers.append(block.attention.S)
        x = self.classifier(x)
        return x.squeeze(2)

    def layer_association_discrepancy(self, Pl, Sl):
        rowwise_kl = lambda row: (
            F.kl_div(Pl[row, :], Sl[row, :]) + F.kl_div(Sl[row, :], Pl[row, :])
        )
        ad_vector = torch.concat(
            [rowwise_kl(row).unsqueeze(0) for row in range(Pl.shape[0])]
        )
        return ad_vector

    def association_discrepancy(self, P_list, S_list):

        return (1 / len(P_list)) * sum(
            [
                self.layer_association_discrepancy(P, S)
                for P, S in zip(P_list, S_list)
            ]
        )

    def loss_function(self, x_hat, P_list, S_list, lambda_, x):
        # frob_norm = torch.linalg.matrix_norm(x_hat - x, ord="fro")
        mse_loss = F.mse_loss(x_hat, x)
        return mse_loss - (
            lambda_
            # * torch.linalg.norm(self.association_discrepancy(P_list, S_list), ord=1)
            * torch.mean(torch.abs(self.association_discrepancy(P_list, S_list)))
        )

    def min_loss(self, preds, y):
        P_list = self.P_layers
        S_list = [S.detach() for S in self.S_layers]
        lambda_ = -self.lambda_
        return self.loss_function(preds, P_list, S_list, lambda_, y)

    def max_loss(self, preds, y):
        P_list = [P.detach() for P in self.P_layers]
        S_list = self.S_layers
        lambda_ = self.lambda_
        return self.loss_function(preds, P_list, S_list, lambda_, y)

    def loss_fn(self, preds, y):
        loss = self.min_loss(preds, y)
        loss.backward(retain_graph=True)
        loss = self.max_loss(preds, y)
        return loss

    def anomaly_score(self, preds, y):
        """
        Not used. Useful in unsupervsied setting but unecessary for our purpose
        """
        ad = F.softmax(
            -self.association_discrepancy(self.P_layers, self.S_layers, y), dim=0
        )
        assert ad.shape[0] == self.N
        norm = torch.tensor(
            [
                torch.linalg.norm(y[i, :] - preds[i, :], ord=2)
                for i in range(self.N)
            ]
        )
        assert norm.shape[0] == self.N
        score = torch.mul(ad, norm)
        return score