from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

def process_data(path, save=False, cooldown_minutes=120, segment_length=50, undersample_factor=100, min_ratio=0.5):
    '''
    path: path of .csv.gz data file
    save: do we cache this file for later use or not? not recommended if you have low space lmaoo
    add_extra_ones: number of minutes worth of rows to change gt to 1 after an anomaly
    '''
    data = pd.read_csv(path, compression='gzip', parse_dates=['date']).drop(columns=['symbol'])
    if cooldown_minutes > 0:
        idxs = data.index[data['gt'] == 1].tolist() # indices where data=1
        for idx in idxs:
            start_date = pd.to_datetime(data['date'].iloc[idx])
            after_pump = data['date'] > start_date
            before_threshold = data['date'] < start_date+pd.Timedelta(minutes=cooldown_minutes)
            data.loc[after_pump & before_threshold, 'gt'] = 2
    data = data.drop(columns=['date'])

    # separate out pumps
    n_pumps = np.max(data['pump_index'].values)
    longest_pump_length = 0
    pumps = []
    for i in range(n_pumps):
        pump_i = data[data['pump_index'] == i]
        longest_pump_length = max(pump_i.shape[0], longest_pump_length)
        pumps.append(pump_i.values[:, 1:])  # remove pump index
    
    # ensure all pumps are same length
    segments = []
    for pump in pumps:
        if pump.shape[0] < segment_length:
            continue
        for i, window in enumerate(sliding_window_view(pump, segment_length, axis=0)):
            window = window.transpose()
            if np.count_nonzero(window[:, -1] != 2) < min_ratio*segment_length:
                continue
            if 1 in window[:, -1]:
                segments.append(window)
            else:
                if i % undersample_factor == 0:
                    segments.append(window)
    pumps = np.stack(segments)

    # save datasets if asked
    if save:
        cached_file_path = f'{path}_{cooldown_minutes}.pkl'
        with open(cached_file_path, 'wb') as f:
            np.save(f, pumps)

def read_data(path, TRAIN_RATIO=0.5, BATCH_SIZE=8, cooldown_minutes=120, save=False):
    '''
    path: path of .csv.gz data file
    TRAIN_RATIO: ratio of data to be used for training -- rest will be held out for testing
    BATCH_SIZE: batch size for training
    add_extra_ones: number of minutes worth of rows to change gt to 1 after an anomaly
    '''
    assert os.path.exists(path)
    cached_file_path = f'{path}_{cooldown_minutes}.pkl'
    if not os.path.exists(cached_file_path):
        process_data(path, save=save, cooldown_minutes=cooldown_minutes)

    # load processed data file
    with open(cached_file_path, 'rb') as f:
        pumps = np.load(f)
    
    # split into train/validate; return dataloaders for each set
    train_data, test_data = train_test_split(pumps, train_size=TRAIN_RATIO)
    train_data, test_data = torch.FloatTensor(train_data), torch.FloatTensor(test_data)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
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
