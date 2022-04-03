from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import os
from torch.utils.data import DataLoader

def process_data(path, save=False):
    data = pd.read_csv(path, compression='gzip').drop(columns=['symbol', 'date'])

    # separate out pumps
    n_pumps = np.max(data['pump_index'].values)
    longest_pump_length = 0
    pumps = []
    for i in range(n_pumps):
        pump_i = data[data['pump_index'] == i]
        longest_pump_length = max(pump_i.shape[0], longest_pump_length)
        pumps.append(pump_i.values)
    
    # ensure all pumps are same length
    for i in range(len(pumps)):
        pumps[i] = np.pad(pumps[i], pad_width=((longest_pump_length-pumps[i].shape[0], 0), (0, 0)))
    pumps = np.stack(pumps)

    # save datasets if asked
    if save:
        with open(path+'.pkl', 'wb') as f:
            np.save(f, pumps)

def read_data(path, TRAIN_RATIO=0.5, BATCH_SIZE=8):
    assert os.path.exists(path)
    if not os.path.exists(path+'.pkl'): process_data(path, save=True)

    # load processed data file
    with open(path+'.pkl', 'rb') as f:
        pumps = np.load(f)
    
    # split into train/validate; return dataloaders for each set
    train_data, test_data = train_test_split(pumps, train_size=TRAIN_RATIO)
    train_data, test_data = torch.FloatTensor(train_data), torch.FloatTensor(test_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)
    return train_loader, test_loader


if __name__ == '__main__':
    '''
    tests data read function
    '''
    process_data('./features_5S.csv.gz', save=True)
    process_data('./features_15S.csv.gz', save=True)
    process_data('./features_25S.csv.gz', save=True)

    # test to make sure we can open it
    read_data('./features_5S.csv.gz')
    read_data('./features_15S.csv.gz')
    read_data('./features_25S.csv.gz')
