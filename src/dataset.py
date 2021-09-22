# functions to process raw data into dataset


from json import load
from tensorflow.keras.utils import timeseries_dataset_from_array, normalize
from tensorflow.python.ops.gen_dataset_ops import window

class Dataset:

    def __init__(self, raw, csv=True, window_size=50, split_ratio=[0.8, 0.1, 0.1]) -> None:
        self.raw = raw
        self.window_size = window_size
        self.dataset_args = {
            'sequence_length': window_size,
            'sequence_stride': 1,
            'sampling_rate': 1,
            'shuffle': True,
            'batch_size': 32,
        }
        self.split_ratio = split_ratio
        self.data, self.targets = self.data_targets(raw, csv)
        self.train, self.validation, self.test = self.split()

    def preprocess(self, raw, csv):
        """ drop the oldest data point and drop the date column"""
        data = raw[::-1].drop(index=raw.index[0],
                            axis=0)  # reverse and remove first day
        if csv:
            data.drop('date', axis=1, inplace=True)  # remove date column
        return data

    def data_targets(self, raw, csv):
        """ raw to dataset """
        normal = self.preprocess(raw, csv=csv)  # open, high, low, close, volume
        normal = normalize(normal, axis=0, order=2)
        data = normal[:-self.window_size]  # the last window_size entries has no target
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

    def split(self):
        """ split dataset into train and test """
        train_end = int(self.split_ratio[0] * len(self.data))
        val_end = train_end + int(self.split_ratio[1] * len(self.data))
        train = self.to_dataset(0, train_end)
        validation = self.to_dataset(train_end+1, val_end)
        test = self.to_dataset(val_end+1, -1)
        return train, validation, test


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
