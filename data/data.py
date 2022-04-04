from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader


def create_loaders(pumps, TRAIN_RATIO, BATCH_SIZE):
    '''
    creates train and test loaders given a list of np-array pumps of equal length
    '''
    # split into train/validate; return dataloaders for each set
    train_data, test_data = train_test_split(pumps, train_size=TRAIN_RATIO)
    train_data, test_data = torch.FloatTensor(train_data), torch.FloatTensor(test_data)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, drop_last=True)
    return train_loader, test_loader

def process_data(path, save=False, add_extra_ones=120):
    '''
    path: path of .csv.gz data file
    save: do we cache this file for later use or not? not recommended if you have low space lmaoo
    add_extra_ones: number of minutes worth of rows to change gt to 1 after an anomaly
    '''
    data = pd.read_csv(path, compression='gzip', parse_dates=['date']).drop(columns=['symbol'])
    if add_extra_ones > 0:
        idxs = data.index[data['gt'] == 1].tolist() # indices where data=1
        for idx in idxs:
            start_date = pd.to_datetime(data['date'].iloc[idx])
            after_pump = data['date'] > start_date
            before_threshold = data['date'] < start_date+pd.Timedelta(minutes=add_extra_ones)
            data.loc[after_pump & before_threshold, 'gt'] = 1
    data = data.drop(columns=['date'])

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
        cached_file_path = f'{path}_{add_extra_ones}.pkl'
        with open(cached_file_path, 'wb') as f:
            np.save(f, pumps)
    else:
        return pumps

def read_data(path, TRAIN_RATIO=0.5, BATCH_SIZE=8, add_extra_ones=120, save=False):
    '''
    path: path of .csv.gz data file
    TRAIN_RATIO: ratio of data to be used for training -- rest will be held out for testing
    BATCH_SIZE: batch size for training
    add_extra_ones: number of minutes worth of rows to change gt to 1 after an anomaly
    '''
    assert os.path.exists(path)
    cached_file_path = f'{path}_{add_extra_ones}.pkl'
    if not os.path.exists(cached_file_path):
        pumps = process_data(path, save=save, add_extra_ones=add_extra_ones)

    # load processed data file
    if not pumps:
        with open(cached_file_path, 'rb') as f:
            pumps = np.load(f)
    
    # return loaders
    return create_loaders(pumps, TRAIN_RATIO, BATCH_SIZE)

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
