from sklearn import preprocessing
import numpy as np

from moving_avg import sma
from config import WIN_SIZE

def preprocess(raw, csv=False):
    """ drop the oldest data point and drop the date column"""
    data = raw.drop(index=raw.index[-1], axis=0)  # remove first day
    if csv:
        data.drop('date', axis=1, inplace=True)  # remove date column
    return data

def normalize(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

def simple_dataset(raw):
    return normalize(preprocess(raw))

def calc_ohlcv(normal, window_size):
    """ normalized open, high, low, close, volume data """
    return np.array([normal[i:i+window_size] for i in range(len(normal)-window_size)])

def calc_indicators(ohlcv):
    return normalize(np.array([sma(window) for window in ohlcv]))

def calc_open(data, window_size):
    next_open = np.array([data[:,0][i+window_size] for i in range(len(data)-window_size)])
    return np.expand_dims(next_open, -1)

def raw_to_dataset(raw, window_size=WIN_SIZE):
    """ raw to dataset """
    data = preprocess(raw)
    normal = normalize(data)

    ohlcv = calc_ohlcv(normal, window_size)
    indicators = calc_indicators(ohlcv)
    open_values = calc_open(data, window_size)  # open value of next day
    open_normal = calc_open(normal, window_size)
    y_normalizer = preprocessing.MinMaxScaler().fit(open_values)

    return ohlcv, indicators, open_normal, open_values, y_normalizer  # opens are next-day
