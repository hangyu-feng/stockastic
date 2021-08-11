from sklearn import preprocessing
import numpy as np

HISTORY_POINTS = 50

def preprocess(data):
    """ drop the oldest data point and drop the date column"""
    data = data.drop('date', axis=1)
    data = data.drop(0, axis=0)
    return data.values

def normalize(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

def calc_ema(values, time_period):
    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = np.mean(values[:,3])
    ema_values = [sma]
    k = 2 / (1 + time_period)
    for i in range(len(his) - time_period, len(his)):
        close = his[i][3]
        ema_values.append(close * k + ema_values[-1] * (1 - k))
    return ema_values[-1]

def process(data, size):
    """ohlcv means open, high, low, close, volume
    data: raw data, without preprocess and normalize
    """
    data = preprocess(data)

    data_normal = normalize(data)
    ohlcv_normal = np.array(data_normal[i:i+size] for i in range(len(data_normal) - size))
    next_day_open_normal = np.array(data_normal[:,0][i + size] for i in range(len(data_normal) - size))
    next_day_open_normal = np.expand_dims(next_day_open_normal, -1)

    next_day_open = np.array(data[:,0][i + size] for i in range(len(data) - size))
    next_day_open = np.expand_dims(next_day_open, -1)

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit(np.expand_dims( next_day_open ))
    # TODO: split into 4 functions


if __name__ == "__main__":
    import pandas
    data = pandas.read_csv("data/daily/AMD.csv")
    pass
