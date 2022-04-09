from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

from data.data import get_data
from models.conv_lstm import ConvLSTM
from models.anomaly_transformer import AnomalyTransformer
from models.utils import count_parameters

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# conv model hyperparameters
EMBEDDING_SIZE = 32
N_LAYERS = 5
N_EPOCHS = 100
KERNEL_SIZE = 5

# train hyperparameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
P_R_THRESHOLD = 0.8  # precision-recall threshold
TRAIN_RATIO = 0.8  # [0.0, 1.0] proportion of data used for train
UNDERSAMPLE_RATIO = 0.05  # [0.0, 1.0] the fraction of data without anomalies to keep during training
SEGEMENT_LENGTH = 60  # number of chunks in a segment


'''
SAVE=True caches a copy of the data for each time you run it with a different ADD_EXTRA_ONES setting.
makes for faster experiments and lower data loading times, but it's not too much faster and each
copy of the data is around 400mb. so up to you.
'''
SAVE = True


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
        preds = model(x).reshape(y.shape)  # goes from (batch_size, seq_len, 1) -> (batch_size, seq_len)
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
            preds = model(x)[:, -1].reshape(y.shape)
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            preds_0.extend(preds[y == 0])
            preds_1.extend(preds[y == 1])
            all_ys.append(y)
            all_preds.append(preds)
    print('Mean anomaly score for 0:', (sum(preds_0) / len(preds_0)).item())
    print('Mean anomaly score for 1:', (sum(preds_1) / len(preds_1)).item())
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
        segment_length=SEGEMENT_LENGTH,
        save=SAVE
    )
    n_feats = train_loader.dataset[0].shape[1] - 1  # since the last column is the target value
    conv_model = ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS).to(device)

    optimizer = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss().to(device)
    models = [conv_model]
    for model in models:
        print(f'model {type(model)} using {count_parameters(model)} parameters.')
        for epoch in range(N_EPOCHS):
            loss = train(model, train_loader, optimizer, criterion, device)
            print(f'Epoch: {epoch}')
            print(f'Train -- Loss: {loss:0.5f}')
            if (epoch + 1) % 5 == 0:
                acc, f1, recall, precision = validate(model, test_loader, device)
                print(
                    f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')
            print()
