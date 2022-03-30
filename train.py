import torch
from torch.utils.data import DataLoader
from models.conv_lstm import *
import pandas as pd


def train(model, iterator, labels, opt, criterion):
    epoch_loss = 0
    for batch, label in iterator, labels:
        opt.zero_grad()
        pred = model(batch)
        loss = criterion(pred, label)
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        print("Batch Loss")
    return epoch_loss / len(iterator)


def main():
    data = pd.read_csv('data/features_5S.csv.gz', compression='gzip')
    labels = pd.read_csv('data/pump_telegram.csv')

    EMBEDDING_SIZE = 64
    N_LAYERS = 1
    LEARNING_RATE = .01
    N_EPOCHS = 10
    KERNEL_SIZE = 2
    BATCH_SIZE = 10

    train_loader = DataLoader(data, batch_size = BATCH_SIZE)
    labels_loader = DataLoader(labels, batch_size=1)

    model = ConvLSTM(16, KERNEL_SIZE, EMBEDDING_SIZE, N_LAYERS)
    opt = torch.optim.Adam(model.params)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(N_EPOCHS):
        print('Epoch: ' + str(epoch))
        train_loss = train(model, train_loader, labels_loader, opt, criterion)
        print('ELoss: ' + str(train_loss))
