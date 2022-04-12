import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import torch
import numpy as np
import random
import argparse
from data.data import create_loader, get_data
from models.conv_lstm import ConvLSTM
from models.anomaly_transformer import AnomalyTransformer
from models.utils import count_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


###   cli arguments   ###
args = argparse.ArgumentParser()
# conv model
args.add_argument('--embedding_size', type=int, default=256)
args.add_argument('--n_layers', type=int, default=1)
args.add_argument('--n_epochs', type=int, default=70)
args.add_argument('--kernel_size', type=int, default=3)
args.add_argument('--dropout', type=float, default=0.0)
args.add_argument('--cell_norm', type=bool, default=False) # bools are weird with argparse. deal with this later
args.add_argument('--out_norm', type=bool, default=False)
# training
args.add_argument('--lr', type=float, default=1e-3)
args.add_argument('--batch_size', type=int, default=256)
args.add_argument('--train_ratio', type=float, default=0.8)
args.add_argument('--undersample_ratio', type=float, default=0.025)
args.add_argument('--segment_length', type=int, default=10)
# validation
args.add_argument('--prthreshold', type=float, default=0.7)
args.add_argument('--kfolds', type=int, default=5)
# ease of use
args.add_argument('--save', type=bool, default=True)
args.add_argument('--validate_every_n', type=int, default=10)
args.add_argument('--train_output_every_n', type=int, default=5)
args.add_argument('--time_epochs', type=bool, default=True)
args.add_argument('--final_run', type=bool, default=False)
config = args.parse_args()


###   hyperparameters   ###
# conv model
EMBEDDING_SIZE = config.embedding_size
N_LAYERS = config.n_layers
N_EPOCHS = config.n_epochs
KERNEL_SIZE = config.kernel_size
DROPOUT = config.dropout
CELL_NORM = config.cell_norm
OUT_NORM = config.out_norm
# training
LEARNING_RATE = config.lr
BATCH_SIZE = config.batch_size
TRAIN_RATIO = config.train_ratio  # [0.0, 1.0] proportion of data used for train
UNDERSAMPLE_RATIO = config.undersample_ratio  # [0.0, 1.0] fraction of majority class samples to keep for training
SEGMENT_LENGTH = config.segment_length  # number of chunks in a segment
# validation
P_R_THRESHOLD = config.prthreshold # precision-recall threshold
K_FOLDS = config.kfolds
# ease of use
SAVE = config.save # saves cache of data, uses ~1-2gb for each choice of segment length
EVERY_N_VALID = config.validate_every_n # run validation every EVERY_N epochs
EVERY_N_TRAIN = config.train_output_every_n
TIME_EPOCHS = config.time_epochs # time each epoch
IS_FINAL_RUN = config.final_run
# reproducability
SEED = 42069
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


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


def validate(model, dataloader, device, verbose=True):
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
    if verbose:
        print(f'Mean output at 0: {(sum(preds_0) / len(preds_0)).item():0.5f} at 1: {(sum(preds_1) / len(preds_1)).item():0.5f}')
    y = torch.cat(all_ys, dim=0).cpu()
    preds = torch.cat(all_preds, dim=0).cpu()
    preds = preds > P_R_THRESHOLD
    accuracy = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    recall = recall_score(y, preds, zero_division=0)
    precision = precision_score(y, preds, zero_division=0)
    return accuracy, precision, recall, f1

def create_conv_model():
    return ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS, 
        cell_norm=CELL_NORM, out_norm=OUT_NORM).to(device)

if __name__ == '__main__':
    #train_loader, test_loader = 
    data = get_data(
        './data/features_15S.csv.gz',
        batch_size=BATCH_SIZE,
        train_ratio=TRAIN_RATIO,
        undersample_ratio=UNDERSAMPLE_RATIO,
        segment_length=SEGMENT_LENGTH,
        save=SAVE
    )
    n_feats = data.shape[-1] - 1 # -1 since last column is the target value
    criterion = torch.nn.BCELoss().to(device)
    models = [create_conv_model]
    kf = KFold(n_splits=K_FOLDS)
    for model_creator in models:
        fold_metrics = np.array([0.0]*4)
        print(f'Model {type(model_creator())} using {count_parameters(model_creator())} parameters:')
        for fold_i, (train_indices, test_indices) in enumerate(kf.split(data)):
            best_metrics = np.array([0.0]*4)
            print(f'#####  fold {fold_i+1}  #####')
            # make model
            model = model_creator()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            # get data
            train_data, test_data = data[train_indices], data[test_indices]
            train_loader = create_loader(train_data, batch_size=BATCH_SIZE, 
                undersample_ratio=UNDERSAMPLE_RATIO, shuffle=True, drop_last=True)
            test_loader = create_loader(test_data, batch_size=BATCH_SIZE)
            for epoch in range(N_EPOCHS):
                start = time.time()
                loss = train(model, train_loader, optimizer, criterion, device)
                if (epoch + 1) % EVERY_N_TRAIN == 0:
                    print(f'Epoch {epoch + 1}{f" ({(time.time()-start):0.2f}s)" if TIME_EPOCHS else ""} -- Train Loss: {loss:0.5f}')
                if (epoch + 1) % EVERY_N_VALID == 0 or IS_FINAL_RUN:
                    acc, precision, recall, f1 = validate(model, test_loader, device, verbose=False)
                    if f1 > best_metrics[-1]:
                        best_metrics = [acc, precision, recall, f1]
                    print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
            fold_metrics += np.array(best_metrics)
            print()
        acc, precision, recall, f1 = fold_metrics / K_FOLDS
        print(f'Final metrics for model {type(model_creator())} ({K_FOLDS} folds)')
        print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
        
        
