# functions to process raw data into dataset


from json import load
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.python.ops.gen_dataset_ops import window


class Dataset:

    def __init__(self, raw, csv=True, window_size=50, split_ratio=[0.7, 0.2, 0.1], sequence_stride=1, sampling_rate=1, shuffle=False, batch_size=32) -> None:
        self.raw = raw
        self.df = preprocess(raw, csv)
        # self.window_size = window_size
        # self.dataset_args = {
        #     'sequence_length': window_size,
        #     'sequence_stride': sequence_stride,
        #     'sampling_rate': sampling_rate,
        #     'shuffle': shuffle,
        #     'batch_size': batch_size,
        # }

        std = split(self.df, split_ratio)['train'].std()
        normalized_df = normalize(self.df, std, span=100)
        self.normalized = split(normalized_df, split_ratio)  # {'train', 'validation', 'test'}

        self.data, self.targets = self.data_targets(raw, csv)
        self.train, self.validation, self.test = self.split(
            split_ratio, self.data)

    def data_targets(self, raw, csv):
        """ raw to dataset """
        normal = self.preprocess(
            raw, csv=csv)  # open, high, low, close, volume
        normal = normalize(normal, axis=0, order=2)
        # the last window_size entries has no target
        data = normal[:-self.window_size]
        targets = normal[self.window_size:][normal.columns[0]]
        return data, targets

    def to_dataset(self, start_index=0, end_index=-1):
        if end_index == -1:
            end_index = len(self.data) - 1
        ds = timeseries_dataset_from_array(
            data=self.data,
            targets=self.targets,
            start_index=start_index,
            end_index=end_index,
            **self.dataset_args
        )
        return ds

    def split(self, split_ratio, data):
        """ split dataset into train and test """
        train_end = int(split_ratio[0] * len(data))
        val_end = train_end + int(split_ratio[1] * len(data))
        train = self.to_dataset(0, train_end)
        validation = self.to_dataset(train_end+1, val_end)
        test = self.to_dataset(val_end+1, -1)
        return train, validation, test

    def describe(self):
        return self.df.describe()


def preprocess(raw, csv):
    """ drop the oldest data point and drop the date column"""
    data = raw[::-1].drop(index=raw.index[0],
                          axis=0)  # reverse and remove first day
    if csv:
        data.drop('date', axis=1, inplace=True)  # remove date column
    return data


def split(df, split_ratio):
    if len(split_ratio) != 3 or sum(split_ratio) != 1:
        raise("split ratio should be a 3-item list and sums to 1")
    n = len(df)
    split1 = int(n * split_ratio[0])
    split2 = int(n * (1 - split_ratio[2]))
    train_df = df[0:split1]
    val_df = df[split1:split2]
    test_df = df[split2:]
    return {"train": train_df, "validation": val_df, "test": test_df}


def normalize(df, std, com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0, times=None):
    """ ewm (exponential moving average) normalization
    df: the dataframe
    std: standard deviation
    """
    # TODO: not sure if this result should be divided by df.std(): return result / df.std()
    ema = df.ewm(com, span, halflife, alpha, min_periods,
                 adjust, ignore_na, axis, times).mean()
    return (df - ema) / std


if __name__ == "__main__":
    symbol = "AMD"
    from data_loader import DataLoader
    loader = DataLoader()
    raw = loader.get_raw(symbol)
    ds = Dataset(raw)

    print("...")
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
