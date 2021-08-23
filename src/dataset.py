from sklearn import preprocessing
from tensorflow.keras.utils import timeseries_dataset_from_array
from config import WIN_SIZE, BATCH_SIZE, SPLIT_RATIO

def preprocess(raw, csv=False):
    """ drop the oldest data point and drop the date column"""
    data = raw[::-1].drop(index=raw.index[0], axis=0)  # reverse and remove first day
    if csv:
        data.drop('date', axis=1, inplace=True)  # remove date column
    return data

def normalize(data):
    return preprocessing.MinMaxScaler().fit_transform(data)

# def simple_dataset(raw):
#     return normalize(preprocess(raw))

# def calc_ohlcv(normal, window_size):
#     """ normalized open, high, low, close, volume data """
#     return np.array([normal[i:i+window_size] for i in range(len(normal)-window_size)])

# def calc_indicators(ohlcv):
#     return normalize(np.array([sma(window) for window in ohlcv]))

# def calc_open(data, window_size):
#     next_open = np.array([data[:,0][i+window_size] for i in range(len(data)-window_size)])
#     return np.expand_dims(next_open, -1)

def raw_to_dataset(raw, window_size=WIN_SIZE, batch_size=BATCH_SIZE):
    """ raw to dataset """
    data = preprocess(raw)  # open, high, low, close, volume
    normal = normalize(data)

    ds = timeseries_dataset_from_array(
        data=normal[:-window_size],
        targets=data[data.columns[0]][window_size:],  # targets are open values
        sequence_length=window_size,
        batch_size=batch_size
    )

    return ds

def split(dataset, ratio=SPLIT_RATIO):
    assert sum(ratio) == 1 and all(i >= 0 for i in ratio)
    train_size = int(ratio[0] * len(dataset))
    val_size = int(ratio[1] * len(dataset))
    train = dataset.take(train_size)
    validation = dataset.skip(train_size).take(val_size)
    test = dataset.skip(train_size + val_size)
    return train, validation, test
