import math
from traceback import print_list

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class AnomalyAttention(nn.Module):
    def __init__(self, N, d_model, device):
        super(AnomalyAttention, self).__init__()
        self.d_model = d_model
        self.N = N
        self.device = device

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Ws = nn.Linear(d_model, 1, bias=False) # sigma

        self.Q = self.K = self.V = self.sigma = torch.zeros((N, d_model))

        self.P = torch.zeros((N, N))
        self.S = torch.zeros((N, N))

    def forward(self, x):

        self.initialize(x)
        self.P = self.prior_association()
        self.S = self.series_association()
        Z = self.reconstruction()

        return Z.squeeze(2)

    def initialize(self, x):
        self.Q = self.Wq(x)
        self.K = self.Wk(x)
        self.V = self.Wv(x)
        self.sigma = self.Ws(x)

    @staticmethod
    def gaussian_kernel(mean, sigma, device):
        normalize = (1 / (math.sqrt(2 * torch.pi) * sigma)).to(device)
        return normalize * torch.exp(-0.5 * (mean.to(device) / sigma.to(device)).pow(2))

    def prior_association(self):
        p = torch.from_numpy(
            np.abs(np.indices((self.N, self.N))[0] - np.indices((self.N, self.N))[1])
        )
        gaussian = self.gaussian_kernel(p.float(), self.sigma, self.device)
        gaussian /= gaussian.sum(dim=(2,1)).unsqueeze(1).unsqueeze(2)
        return gaussian

    def series_association(self):
        return F.softmax(torch.bmm(self.Q, self.K.transpose(1,2)) / math.sqrt(self.d_model), dim=0)

    def reconstruction(self):
        return self.S @ self.V


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


class AnomalyTransformer(nn.Module):
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