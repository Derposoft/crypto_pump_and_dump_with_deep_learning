import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import numpy as np
import random
import argparse
from data.data import get_data
from models.conv_lstm import ConvLSTM
from models.anomaly_transformer import AnomalyTransformer
from models.utils import count_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###   cli arguments   ###
args = argparse.ArgumentParser()
# conv model
args.add_argument('--embedding_size', type=int, default=16)
args.add_argument('--n_layers', type=int, default=5)
args.add_argument('--n_epochs', type=int, default=100)
args.add_argument('--kernel_size', type=int, default=5)
args.add_argument('--dropout', type=float, default=0.0)
args.add_argument('--layernorm', type=bool, default=False)
# training
args.add_argument('--lr', type=float, default=1e-3)
args.add_argument('--batch_size', type=int, default=32)
args.add_argument('--prthreshold', type=float, default=0.9)
args.add_argument('--train_ratio', type=float, default=0.5)
args.add_argument('--undersample_ratio', type=float, default=0.05)
args.add_argument('--segment_length', type=int, default=5)
# ease of use
args.add_argument('--save', type=bool, default=True)
args.add_argument('--validate_every_n', type=int, default=10)
args.add_argument('--time_epochs', type=bool, default=True)
config = args.parse_args()


###   hyperparameters   ###
# conv model
EMBEDDING_SIZE = config.embedding_size
N_LAYERS = config.n_layers
N_EPOCHS = config.n_epochs
KERNEL_SIZE = config.kernel_size
DROPOUT = config.dropout
LAYERNORM = config.layernorm
# training
LEARNING_RATE = config.lr
BATCH_SIZE = config.batch_size
P_R_THRESHOLD = config.prthreshold # precision-recall threshold
TRAIN_RATIO = config.train_ratio  # [0.0, 1.0] proportion of data used for train
UNDERSAMPLE_RATIO = config.undersample_ratio  # [0.0, 1.0] fraction of majority class samples to keep for training
SEGMENT_LENGTH = config.segment_length  # number of chunks in a segment
# ease of use
SAVE = config.save # saves cache of data, uses ~1-2gb for each choice of segment length
EVERY_N = config.validate_every_n # run validation every EVERY_N epochs
TIME_EPOCHS = config.time_epochs # time each epoch
# reproducability
SEED = 42069
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


def train(model, dataloader, opt, criterion, device):
    '''
    trains given model with given dataloader, optimizer, criterion, and on device.
    :returns: loss
    '''
    epoch_loss = 0
    for batch in dataloader:
        # training step
        opt.zero_grad()
        x = batch[:, :, :-1].to(device)
        y = batch[:, :, -1].to(device)
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


def validate(model, dataloader, device):
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
            preds_0.extend(preds[y == 0])
            preds_1.extend(preds[y == 1])
            all_ys.append(y)
            all_preds.append(preds)
    print(f'Mean output at 0: {(sum(preds_0) / len(preds_0)).item():0.5f} at 1: {(sum(preds_1) / len(preds_1)).item():0.5f}')
    y = torch.cat(all_ys, dim=0).cpu()
    preds = torch.cat(all_preds, dim=0).cpu()
    preds = preds > P_R_THRESHOLD
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    return accuracy, f1, recall, precision


if __name__ == '__main__':
    train_loader, test_loader = get_data(
        './data/features_15S.csv.gz',
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        undersample_ratio=UNDERSAMPLE_RATIO,
        segment_length=SEGMENT_LENGTH,
        save=SAVE
    )
    n_feats = train_loader.dataset[0].shape[1] - 1  # since the last column is the target value
    conv_model = ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS, norm=LAYERNORM).to(device)

    optimizer = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss().to(device)
    models = [conv_model]
    for model in models:
        print(f'model {type(model)} using {count_parameters(model)} parameters.')
        for epoch in range(N_EPOCHS):
            start = time.time()
            loss = train(model, train_loader, optimizer, criterion, device)
            print(f'Epoch {epoch + 1}{f" ({(time.time()-start):0.2f}s)" if TIME_EPOCHS else ""} -- Train Loss: {loss:0.5f}')
            if (epoch + 1) % EVERY_N == 0:
                acc, f1, recall, precision = validate(model, test_loader, device)
                print(
                    f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
                print()
