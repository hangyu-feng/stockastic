# functions to process raw data into dataset


from json import load
from tensorflow.keras.utils import timeseries_dataset_from_array, normalize
from tensorflow.python.ops.gen_dataset_ops import window

class Dataset:

    def __init__(self, raw, csv=True, window_size=50) -> None:
        self.raw = raw
        self.window_size = window_size
        self.dataset_args = {
            'sequence_length': window_size,
            'sequence_stride': 1,
            'sampling_rate': 1,
            'shuffle': True,
            'batch_size': 32,
        }
        # a tf.data.Dataset object
        self.ds = self.raw_to_dataset(raw, csv=csv)

    def preprocess(self, raw, csv):
        """ drop the oldest data point and drop the date column"""
        data = raw[::-1].drop(index=raw.index[0],
                            axis=0)  # reverse and remove first day
        if csv:
            data.drop('date', axis=1, inplace=True)  # remove date column
        return data

    def raw_to_dataset(self, raw, csv):
        """ raw to dataset """
        normal = self.preprocess(raw, csv=csv)  # open, high, low, close, volume
        normal = normalize(normal, axis=0, order=2)
        window_size = self.dataset_args['window_size']
        data = normal[:-window_size]  # the last window_size entries has no target
        targets = normal[window_size:][normal.columns[0]]
        ds = timeseries_dataset_from_array(
            data=data,
            targets=targets,
            **self.dataset_args
        )
        return ds

    def split(self, dataset, ratio=SPLIT_RATIO):
        """ split dataset into train and test """
        assert sum(ratio) == 1 and all(i >= 0 for i in ratio)
        train_size = int(ratio[0] * len(dataset))
        val_size = int(ratio[1] * len(dataset))
        train = dataset.take(train_size)
        validation = dataset.skip(train_size).take(val_size)
        test = dataset.skip(train_size+val_size)
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
