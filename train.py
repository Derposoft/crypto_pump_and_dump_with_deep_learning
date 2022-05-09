import os
import json
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from data.data import create_loader, create_loaders, get_data
from models.conv_lstm import ConvLSTM
from models.anomaly_transformer import AnomalyTransformer, AnomalyTransfomerIntermediate, AnomalyTransfomerBasic
from models.transformer import TransformerTimeSeries
from models.utils import count_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_metrics_n_epochs(model, *, train_loader, test_loader,
                            optimizer, criterion, device, config, lr_scheduler=None, feature_count=13):
    best_metrics = np.array([0.0]*4)
    for epoch in range(config.n_epochs):
        start = time.time()
        loss = train(model, train_loader, optimizer, criterion, device, feature_count)
        if (epoch + 1) % config.train_output_every_n == 0:
            print(f'Epoch {epoch + 1}{f" ({(time.time()-start):0.2f}s)" if config.time_epochs else ""} -- Train Loss: {loss:0.5f}')
        if (epoch + 1) % config.validate_every_n == 0 or config.final_run:
            if config.prthreshold > 0:
                prthreshold = config.prthreshold
            else:
                prthreshold = pick_threshold(model, train_loader, config.undersample_ratio, device, verbose=config.verbose, feature_count=feature_count)
            acc, precision, recall, f1 = validate(model, test_loader, device, verbose=config.verbose, pr_threshold=prthreshold, feature_count=feature_count)
            if f1 > best_metrics[-1]:
                best_metrics = [acc, precision, recall, f1]
            print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
        if config.lr_decay_step > 0 and (epoch+1) % config.lr_decay_step == 0:
            if lr_scheduler: lr_scheduler.step(epoch+1)
    return best_metrics


def train(model, dataloader, opt, criterion, device, feature_count=13):
    '''
    trains given model with given dataloader, optimizer, criterion, and on device.
    :returns: avg loss/batch for this epoch
    '''
    epoch_loss = 0
    for batch in dataloader:
        opt.zero_grad()
        x = batch[:, :, :feature_count].to(device)
        y = batch[:, :, -1].to(device)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate(model, dataloader, device, verbose=True, pr_threshold=0.7, criterion=None, feature_count=13):
    preds_1 = []
    preds_0 = []
    all_ys = []
    all_preds = []
    epoch_loss = 0
    for batch in dataloader:
        with torch.no_grad():
            # only consider the last chunk of each segment for validation
            x = batch[:, :, :feature_count].to(device)
            y = batch[:, -1, -1].to(device)
            preds = model(x)[:, -1]
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            if verbose:
                preds_0.extend(preds[y == 0])
                preds_1.extend(preds[y == 1])
            all_ys.append(y)
            all_preds.append(preds)
            if criterion is not None:
                loss = criterion(preds, y)
                epoch_loss += loss.item()
    if verbose:
        print(f'Mean output at 0: {(sum(preds_0) / len(preds_0)).item():0.5f} at 1: {(sum(preds_1) / len(preds_1)).item():0.5f}')
    y = torch.cat(all_ys, dim=0).cpu()
    preds = torch.cat(all_preds, dim=0).cpu()
    preds = preds >= pr_threshold
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    if criterion is not None:
        return acc, precision, recall, f1, epoch_loss/len(dataloader)
    else:
        return acc, precision, recall, f1


def pick_threshold(model, dataloader, undersample_ratio, device, verbose=True, feature_count=13):
    all_ys = []
    all_preds = []
    for batch in dataloader:
        with torch.no_grad():
            # only consider the last chunk of each segment for validation
            x = batch[:, :, :feature_count].to(device)
            y = batch[:, -1, -1].to(device)
            preds = model(x)[:, -1]
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            all_ys.append(y)
            all_preds.append(preds)
    y = torch.cat(all_ys, dim=0).cpu()
    preds = torch.cat(all_preds, dim=0).cpu()
    y = y.numpy()
    preds = preds.numpy()
    _, _, thresholds = precision_recall_curve(y, preds)

    best_f1 = 0
    best_threshold = 0
    for threshold in thresholds:
        true_pos = np.sum(preds[y == 1] >= threshold)
        false_pos = np.sum(preds[y == 0] >= threshold)
        false_neg = np.sum(preds[y == 1] < threshold)
        true_neg = np.sum(preds[y == 0] < threshold)

        false_pos /= undersample_ratio
        true_neg /= undersample_ratio

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    if verbose:
        print(f'Best threshold: {best_threshold} (train f1: {best_f1})')

    return best_threshold

def create_conv_model(config):
    return ConvLSTM(n_feats, config.kernel_size, config.embedding_size, config.n_layers, dropout=config.dropout,
        cell_norm=config.cell_norm, out_norm=config.out_norm).to(device)

def create_transformer(config):
    if config.transformer_model == "AnomalyTransformer":
        return AnomalyTransformer(config.segment_length, config.feature_size, config.n_layers, config.lambda_, device).to(device)
    elif config.transformer_model == "TransformerTimeSeries":
        return TransformerTimeSeries(config.feature_size, 1, config.n_head, config.n_layer, config.dropout).to(device)
    elif config.transformer_model == "AnomalyTransfomerIntermediate":
        return AnomalyTransfomerIntermediate(config.segment_length, config.feature_size, config.n_layers, config.lambda_, device).to(device)
    elif config.transformer_model == "AnomalyTransfomerBasic":
        return AnomalyTransfomerBasic(config.segment_length, config.feature_size, config.n_layers, device).to(device)

def parse_args():
    ###   cli arguments   ###
    args = argparse.ArgumentParser()
    args.add_argument('--model', type=str, default='CLSTM', choices=['CLSTM', 'AT'],
        help='Choose CLSTM for the CLSTM model and AT for the Anomaly Transformer model.')
    # conv model
    args.add_argument('--embedding_size', type=int, default=350)
    args.add_argument('--n_layers', type=int, default=1)
    args.add_argument('--n_epochs', type=int, default=100)
    args.add_argument('--kernel_size', type=int, default=3)
    args.add_argument('--dropout', type=float, default=0.0)
    args.add_argument('--cell_norm', type=bool, default=False) # bools are weird with argparse. deal with this later
    args.add_argument('--out_norm', type=bool, default=False)
    # transformer
    args.add_argument('--feature_size', type=int, default=13)
    args.add_argument('--n_head', type=int, default=3)
    args.add_argument('--lambda_', type=float, default=0)
    # training
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--lr_decay_step', type=int, default=0)
    args.add_argument('--lr_decay_factor', type=float, default=0.5)
    args.add_argument('--weight_decay', type=float, default=0.0)
    args.add_argument('--batch_size', type=int, default=1200)
    args.add_argument('--train_ratio', type=float, default=0.8)
    args.add_argument('--undersample_ratio', type=float, default=0.05)
    args.add_argument('--segment_length', type=int, default=15)
    # validation
    args.add_argument('--prthreshold', type=float, default=0.0)
    args.add_argument('--kfolds', type=int, default=1)
    # ease of use
    args.add_argument('--save', type=bool, default=True)
    args.add_argument('--validate_every_n', type=int, default=10)
    args.add_argument('--train_output_every_n', type=int, default=5)
    args.add_argument('--time_epochs', type=bool, default=True)
    args.add_argument('--final_run', type=bool, default=False)
    args.add_argument('--verbose', type=bool, default=False)
    args.add_argument('--dataset', type=str, default='./data/features_25S.csv.gz')
    args.add_argument('--config', type=str, default='')
    args.add_argument('--seed', type=int, default=0xA455) # secret message value
    args.add_argument('--run_count', type=int, default=1)
    return args.parse_args()


if __name__ == '__main__':
    config = parse_args()
    # reproducability
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    g = torch.Generator()
    g.manual_seed(config.seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

    data = get_data(
        config.dataset,
        batch_size=config.batch_size,
        train_ratio=config.train_ratio,
        undersample_ratio=config.undersample_ratio,
        segment_length=config.segment_length,
        save=config.save
    )

    if config.feature_size == -1:
        n_feats = data.shape[-1] - 1 # -1 since last column is the target value
    else:
        n_feat = config.feature_size
    
    criterion = torch.nn.BCELoss().to(device)
    if config.model == "CLSTM":
        models = [create_conv_model] * config.run_count
    else:
        models = [create_transformer] * config.run_count
    
    for model_index, model_creator in enumerate(models):
        if len(models) > 1:
            print(f'Running model {model_index + 1} of {len(models)}')
        fold_metrics = np.array([0.0]*4)
        sample_model = model_creator(config) # used only for debug output in the line below (and a similar line after all folds)
        print(f'Model {type(sample_model)} using {count_parameters(sample_model)} parameters:')
        if config.kfolds > 1:
            kf = KFold(n_splits=config.kfolds)
            for fold_i, (train_indices, test_indices) in enumerate(kf.split(data)):
                print(f'#####  fold {fold_i+1}  #####')
                # make model
                model = model_creator(config)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
                #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_decay_factor, verbose=True, mode='max')
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_factor, verbose=True)
                # create dataloaders and start training loop
                train_data, test_data = data[train_indices], data[test_indices]
                train_loader = create_loader(train_data, batch_size=config.batch_size,
                    undersample_ratio=config.undersample_ratio, shuffle=True, drop_last=True, generator=g)
                test_loader = create_loader(test_data, batch_size=config.batch_size, drop_last=False)
                if config.model == "AnomalyTransfomerIntermediate" or config.model == "AnomalyTransformer":
                    criterion = model.loss_fn
                best_metrics = collect_metrics_n_epochs(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    config=config,
                    lr_scheduler=lr_scheduler,
                    feature_count=config.feature_size,
                )
                fold_metrics += np.array(best_metrics)
                print(f'Best F1 for this fold: {best_metrics[-1]}')
                print()
        else:
            model = model_creator(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            train_loader, test_loader = create_loaders(data, train_ratio=config.train_ratio,
                batch_size=config.batch_size, undersample_ratio=config.undersample_ratio)
            best_metrics = collect_metrics_n_epochs(
                model,
                train_loader=train_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=config
            )
            fold_metrics += np.array(best_metrics)
            print(f'Best F1 this run: {best_metrics[-1]}')
            print()


        acc, precision, recall, f1 = fold_metrics / config.kfolds
        print(f'Final metrics for model {type(sample_model)} ({config.kfolds} folds)')
        print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
