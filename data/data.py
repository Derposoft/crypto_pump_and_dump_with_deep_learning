import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

FEATURE_NAMES = [
    'std_rush_order',
    'avg_rush_order',
    'std_trades',
    'std_volume',
    'avg_volume',
    'std_price',
    'avg_price',
    'avg_price_max',
    'hour_sin',
    'hour_cos',
    'minute_sin',
    'minute_cos',
    'delta_minutes',
]


def load_data(path):
    return pd.read_csv(path, compression='gzip', parse_dates=['date'])


PLACEHOLDER_TIMEDELTA = pd.Timedelta(minutes=0)
MIN_PUMP_SIZE = 100


def get_pumps(data, segment_length, *, pad=True):
    pumps = []
    skipped_row_count = 0
    for pump_index in np.unique(data['pump_index'].values):
        pump_i = data[data['pump_index'] == pump_index].copy()
        if len(pump_i) < MIN_PUMP_SIZE:
            print(f'Pump {pump_index} has {len(pump_i)} rows, skipping')
            skipped_row_count += len(pump_i)
            continue
        pump_i['delta_minutes'] = (pump_i['date'] - pump_i['date'].shift(1)).fillna(PLACEHOLDER_TIMEDELTA)
        pump_i['delta_minutes'] = pump_i['delta_minutes'].apply(lambda x: x.total_seconds() / 60)
        pump_i = pump_i[FEATURE_NAMES + ['gt']]
        pump_i = pump_i.values.astype(np.float32)
        if pad:
            pump_i = np.pad(pump_i, ((segment_length - 1, 0), (0, 0)), 'reflect')
        pumps.append(pump_i)
    print(f'Skipped {skipped_row_count} rows total')
    print(f'{len(pumps)} pumps')
    return pumps


def process_data(data, *, segment_length=60, remove_post_anomaly_data=False):
    print('Processing data...')
    print(f'Segment length: {segment_length}')
    print(f'Remove post anomaly data: {remove_post_anomaly_data}')
    print(f'Data shape: {data.shape}')
    pumps = get_pumps(data, segment_length)
    segments = []
    remove_cnt = 0
    for pump in pumps:
        for i, window in enumerate(np.lib.stride_tricks.sliding_window_view(pump, segment_length, axis=0)):
            segment = window.transpose()
            if remove_post_anomaly_data and segment[:-1, -1].sum() > 0:
                remove_cnt += 1
                continue
            segments.append(segment)
    if remove_post_anomaly_data:
        print(f'Removed {remove_cnt} rows with post-anomaly data')
    print(f'{len(segments)} rows of data after processing')
    return np.stack(segments)


def undersample_train_data(train_data, undersample_ratio):
    with_anomalies = train_data[:, :, -1].sum(axis=1) > 0
    mask = with_anomalies | (np.random.rand(train_data.shape[0]) < undersample_ratio)
    return train_data[mask]


def get_data(path, *,
             train_ratio,
             batch_size,
             undersample_ratio,
             segment_length,
             save=False):
    '''
    path: path of .csv.gz data file
    train_ratio: ratio of data to be used for training by pump index -- rest will be held out for testing
    training_batch_size: batch size for training
    undersample_ratio: ratio of segments without anomalies to keep in training data
    segment_length: length of segments to use in number of chunks (rows)
    '''
    assert os.path.exists(path)

    cached_data_path = f'{path}_{segment_length}.npy'
    if not os.path.exists(cached_data_path):
        data = process_data(load_data(path), segment_length=segment_length)
        if save:
            np.save(cached_data_path, data)
    else:
        print(f'Loading cached data from {cached_data_path}')
        data = np.load(cached_data_path)

    return create_loaders(data, train_ratio=train_ratio, batch_size=batch_size, undersample_ratio=undersample_ratio)


def create_loaders(data, *, train_ratio, batch_size, undersample_ratio):
    '''
    creates train and test loaders given a list of np-array pumps of equal length
    '''
    # split into train/validate; return dataloaders for each set
    train_data, test_data = train_test_split(data, train_size=train_ratio, shuffle=False)
    print(f'Train data shape: {train_data.shape}')
    train_data = undersample_train_data(train_data, undersample_ratio)
    print(f'Train data shape after undersampling: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    print(f'{test_data[:, -1, -1].sum()} segments in test data ending in anomaly')
    train_data, test_data = torch.FloatTensor(train_data), torch.FloatTensor(test_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
    return train_loader, test_loader



if __name__ == '__main__':
    '''
    tests get data function
    '''
    get_data('./features_5S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
    get_data('./features_15S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
    get_data('./features_25S.csv.gz', train_ratio=0.8, batch_size=128, undersample_ratio=0.05, segment_length=60, save=True)
