from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

from data.data import read_data
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
N_SEQ = 100
BATCH_SIZE = 128
P_R_THRESHOLD = 0.5 # precision-recall threshold
TRAIN_RATIO = 0.6 # [0.0, 1.0] proportion of data used for train
COOLDOWN_MINUTES = 30 # labels in [anomaly_time, anomaly_time+COOLDOWN_MINUTES] are ignored in loss
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
        x = batch[:,:,:-1].to(device)
        y = batch[:,:,-1].to(device)
        preds = model(x).reshape(y.shape) # goes from (batch_size, seq_len, 1) -> (batch_size, seq_len)
        loss = criterion(preds, y)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    return epoch_loss/len(dataloader)

def validate(model, dataloader, device):
    accuracy = 0
    f1 = 0
    recall = 0
    precision = 0
    n_batches = len(dataloader)
    means_1 = []
    means_0 = []
    for batch in dataloader:
        with torch.no_grad():
            x = batch[:, :, :-1].to(device)
            y = batch[:, :, -1].to(device)
            raw_preds = model(x).reshape(y.shape)
            preds = (raw_preds > P_R_THRESHOLD).float()
            y, preds = y.cpu().flatten(), preds.cpu().flatten()
            mask = y != 2
            y = y[mask]
            preds = preds[mask]
            means_0.append(raw_preds.flatten()[mask][y==0].mean())
            means_1.append(raw_preds.flatten()[mask][y==1].mean())
            accuracy += accuracy_score(y, preds)
            f1 += f1_score(y, preds, zero_division=1)
            recall += recall_score(y, preds, zero_division=1)
            precision += precision_score(y, preds, zero_division=1)
    print('Mean anomaly score for 0:', sum(means_0)/len(means_0))
    print('Mean anomaly score for 1:', sum(means_1)/len(means_1))
    return accuracy/n_batches, f1/n_batches, recall/n_batches, precision/n_batches


def wrap_loss(loss):
    def wrapped(pred, y):
        mask = y != 2
        return loss(pred[mask], y[mask])

    return wrapped


if __name__ == '__main__':
    train_loader, test_loader = read_data(
        './data/features_15S.csv.gz',
        BATCH_SIZE=BATCH_SIZE,
        TRAIN_RATIO=TRAIN_RATIO,
        cooldown_minutes=COOLDOWN_MINUTES,
        save=SAVE
    )
    n_feats = train_loader.dataset[0].shape[1]-1 # since the last column is the target value
    conv_model = ConvLSTM(n_feats, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS).to(device)

    optimizer = torch.optim.Adam(conv_model.parameters(), lr=LEARNING_RATE)
    criterion = wrap_loss(torch.nn.MSELoss().to(device)) # classic for anomaly detection
    models = [conv_model]
    for model in models:
        print(f'model {type(model)} using {count_parameters(model)} parameters.')
        for epoch in range(N_EPOCHS):
            loss = train(model, train_loader, optimizer, criterion, device)
            acc, f1, recall, precision = validate(model, test_loader, device)
            print(f'Epoch: {epoch}')
            print(f'Train -- Loss: {loss:0.5f}')
            print(f'Val   -- Acc: {acc:0.5f} -- Precision: {precision:0.5f} -- Recall: {recall:0.5f} -- F1: {f1:0.5f}')


