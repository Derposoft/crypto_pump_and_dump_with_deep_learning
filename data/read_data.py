from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

def read_data(path, N_SEQ=100, TRAIN_RATIO=0.5, BATCH_SIZE=32):
    data = pd.read_csv(path, compression='gzip').drop(columns=['symbol', 'date'])

    # separate out pumps
    n_pumps = np.max(data['pump_index'].values)
    longest_pump_length = 0
    pumps = []
    for i in range(n_pumps):
        pump_i = data[data['pump_index'] == i]
        longest_pump_length = max(pump_i.shape[0], longest_pump_length)
        pumps.append(torch.tensor(pump_i.values))
    
    # ensure all pumps are same length
    for pump in pumps:
        pump = F.pad(pump, pad=(0, 0, longest_pump_length-pump.shape[0], 0))

    # split into train/validate; return dataloaders for each set
    train_data, test_data = train_test_split(data, train_size=TRAIN_RATIO)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    '''
    tests data read function
    '''
    read_data('./features_5S.csv.gz')
