import os
import json
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch
import numpy as np
import random
import argparse
from data.data import create_loader, create_loaders, get_data
from models.conv_lstm import ConvLSTM
from models.anomaly_transformer import AnomalyTransformer
from models.utils import count_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collect_metrics_n_epochs(model, *, train_loader, test_loader,
                            optimizer, criterion, device, config, lr_scheduler=None):
    best_metrics = np.array([0.0]*4)
    for epoch in range(config.n_epochs):
        start = time.time()
        loss = train(model, train_loader, optimizer, criterion, device)
        if (epoch + 1) % config.train_output_every_n == 0:
            print(f'Epoch {epoch + 1}{f" ({(time.time()-start):0.2f}s)" if config.time_epochs else ""} -- Train Loss: {loss:0.5f}')
        if (epoch + 1) % config.validate_every_n == 0 or config.final_run:
            acc, precision, recall, f1 = validate(model, test_loader, device, verbose=config.verbose, pr_threshold=config.prthreshold)
            if f1 > best_metrics[-1]:
                best_metrics = [acc, precision, recall, f1]
            print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
        if config.lr_decay_step > 0 and (epoch+1) % config.lr_decay_step == 0:
            if lr_scheduler: lr_scheduler.step(epoch+1)
    return best_metrics

def train(model, dataloader, opt, criterion, device):
    '''
    trains given model with given dataloader, optimizer, criterion, and on device.
    :returns: avg loss/batch for this epoch
    '''
    epoch_loss = 0
    for batch in dataloader:
        opt.zero_grad()
        x = batch[:, :, :-1].to(device)
        y = batch[:, :, -1].to(device)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate(model, dataloader, device, verbose=True, pr_threshold=0.7):
    preds_1 = []
    preds_0 = []
    all_ys = []
    all_preds = []
    for batch in dataloader:
        with torch.no_grad():
            # only consider the last chunk of each segment for validation
            x = batch[:, :, :-1].to(device)
            y = batch[:, -1, -1].to(device)
            preds = model(x)[:, -1]
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            if verbose:
                preds_0.extend(preds[y == 0])
                preds_1.extend(preds[y == 1])
            all_ys.append(y)
            all_preds.append(preds)
    if verbose:
        print(f'Mean output at 0: {(sum(preds_0) / len(preds_0)).item():0.5f} at 1: {(sum(preds_1) / len(preds_1)).item():0.5f}')
    y = torch.cat(all_ys, dim=0).cpu()
    preds = torch.cat(all_preds, dim=0).cpu()
    preds = preds > pr_threshold
    acc = accuracy_score(y, preds)
    precision = precision_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    return acc, precision, recall, f1


def create_conv_model(config):
    return ConvLSTM(n_feats, config.kernel_size, config.embedding_size, config.n_layers, dropout=config.dropout,
        cell_norm=config.cell_norm, out_norm=config.out_norm).to(device)


def parse_args():
    ###   cli arguments   ###
    args = argparse.ArgumentParser()
    # conv model
    args.add_argument('--embedding_size', type=int, default=350)
    args.add_argument('--n_layers', type=int, default=1)
    args.add_argument('--n_epochs', type=int, default=200)
    args.add_argument('--kernel_size', type=int, default=3)
    args.add_argument('--dropout', type=float, default=0.0)
    args.add_argument('--cell_norm', type=bool, default=False) # bools are weird with argparse. deal with this later
    args.add_argument('--out_norm', type=bool, default=False)
    # training
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--lr_decay_step', type=int, default=0)
    args.add_argument('--lr_decay_factor', type=float, default=0.5)
    args.add_argument('--batch_size', type=int, default=800)
    args.add_argument('--train_ratio', type=float, default=0.8)
    args.add_argument('--undersample_ratio', type=float, default=0.05)
    args.add_argument('--segment_length', type=int, default=15)
    # validation
    args.add_argument('--prthreshold', type=float, default=0.7)
    args.add_argument('--kfolds', type=int, default=5)
    # ease of use
    args.add_argument('--save', type=bool, default=True)
    args.add_argument('--validate_every_n', type=int, default=10)
    args.add_argument('--train_output_every_n', type=int, default=5)
    args.add_argument('--time_epochs', type=bool, default=True)
    args.add_argument('--final_run', type=bool, default=False)
    args.add_argument('--verbose', type=bool, default=False)
    args.add_argument('--dataset', type=str, default='./data/features_5S.csv.gz')
    args.add_argument('--config', type=str, default='')
    args.add_argument('--seed', type=int, default=42069)
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
    n_feats = data.shape[-1] - 1 # -1 since last column is the target value
    criterion = torch.nn.BCELoss().to(device)
    models = [create_conv_model]
    for model_creator in models:
        fold_metrics = np.array([0.0]*4)
        sample_model = model_creator(config) # used only for debug output in the line below (and a similar line after all folds)
        print(f'Model {type(sample_model)} using {count_parameters(sample_model)} parameters:')
        if config.kfolds > 1:
            kf = KFold(n_splits=config.kfolds)
            for fold_i, (train_indices, test_indices) in enumerate(kf.split(data)):
                print(f'#####  fold {fold_i+1}  #####')
                # make model
                model = model_creator(config)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
                #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.lr_decay_factor, verbose=True, mode='max')
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_factor, verbose=True)
                # create dataloaders and start training loop
                train_data, test_data = data[train_indices], data[test_indices]
                train_loader = create_loader(train_data, batch_size=config.batch_size, 
                    undersample_ratio=config.undersample_ratio, shuffle=True, drop_last=True, generator=g)
                test_loader = create_loader(test_data, batch_size=config.batch_size, drop_last=False)
                best_metrics = collect_metrics_n_epochs(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    optimizer=optimizer, 
                    criterion=criterion, 
                    device=device, 
                    config=config,
                    lr_scheduler=lr_scheduler
                )
                fold_metrics += np.array(best_metrics)
                print(f'Best F1 for this fold: {best_metrics[-1]}')
                print()
        else:
            model = model_creator(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
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
            print(f'Best F1 this run: {best_metrics[-1]}')

            

        acc, precision, recall, f1 = fold_metrics / config.kfolds
        print(f'Final metrics for model {type(sample_model)} ({config.kfolds} folds)')
        print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
        
        
