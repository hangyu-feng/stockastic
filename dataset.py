from sklearn import preprocessing
from .moving_avg import sma
import numpy as np

def preprocess(raw):
    """ drop the oldest data point and drop the date column"""
    data = raw.drop('date', axis=1)
    data = data.drop(0, axis=0)
    return data.values

def normalize(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

def calc_ohlcv(normal, window_size):
    """ normalized open, high, low, close, volume data """
    return np.array([normal[i:i+window_size] for i in range(len(normal)-window_size)])

def calc_indicators(ohlcv):
    return normalize(np.array([sma(window) for window in ohlcv]))

def calc_open(data, window_size):
    next_open = np.array([data[:,0][i+window_size] for i in range(len(data)-window_size)])
    return np.expand_dims(next_open, -1)

def dataset(raw, window_size=50):
    data = preprocess(raw)
    normal = normalize(data)

    ohlcv = calc_ohlcv(normal, window_size)
    indicators = calc_indicators(ohlcv)
    open_values = calc_open(data, window_size)
    open_normal = calc_open(normal, window_size)
    y_normalizer = preprocessing().MinMaxScaler().fit(open_values)

    return ohlcv, indicators, open_normal, open_values, y_normalizer
