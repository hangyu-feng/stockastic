from alpha_vantage.timeseries import TimeSeries
import json
import pandas
import numpy as np

from dataset import raw_to_dataset, simple_dataset
from config import DATA_PATH, CREDENTIALS_PATH

class DataLoader:
    """ deal with api and file reading/saving """

    def __init__(self, apikey=None) -> None:
        self.apikey = apikey if apikey else self.load_apikey()
        self.ts = TimeSeries(key=self.apikey, output_format='pandas')

    def load_apikey(self, fpath=CREDENTIALS_PATH):
        with open(fpath, 'r') as credentials:
            return json.load(credentials)['alpha_vantage']['api_key']

    def interval_to_method(self, interval):
        method_map = {
            'daily': self.ts.get_daily, 'weekly': self.ts.get_weekly, 'monthly': self.ts.get_monthly,
        }
        if interval in method_map:
            return method_map[interval]
        if interval in {'1min', '5min', '15min', '30min', '60min'}:
            return lambda *args, **kwargs: self.ts.get_intraday(*args, **kwargs, interval=interval),

    def path(self, catogory, symbol, interval):
        """
        catogory: one of "raw", "datasets"
        symbol: (str) stock symbol
        interval: (str) one of 'daily', 'weekly', 'monthly', '1min', '5min', '15min', '30min', '60min'
        """
        return f"{DATA_PATH}/{catogory}/{interval}/{symbol}.pkl"

    def save_timeseries(self, symbol, interval):
        """ pull data from alphavantage.co and save to pickle file """
        get_timeseries = self.interval_to_method(interval)
        data, metadata = get_timeseries(symbol=symbol, outputsize='full')
        data.to_pickle(self.path('raw', symbol, interval))

    def read_timeseries(self, symbol, interval):
        """ read from saved pickle file """
        return pandas.read_pickle(self.path('raw', symbol, interval))

    def get_raw(self, symbol, interval='daily', update=False):
        if update:
            self.save_timeseries(symbol, interval)
        # will throw error if file not found
        return self.read_timeseries(symbol, interval)

    def save_dataset(self, symbol, interval='daily', update=False):
        raw = self.get_raw(symbol, interval, update)
        data = simple_dataset(raw)
        np.save(self.path('datasets', symbol, interval), data)

    def read_dataset(self, symbol, interval='daily'):
        return np.load(self.path('datasets', symbol, interval))

    # ================== below are old methods ==================

    def dataset_OLD(self, symbol, update=False):
        """ grab data from api and return the processed dataset. If update is False, it will only return
        saved data """
        raw = self.get_raw(symbol, update=update)
        return raw_to_dataset(raw)


dl = DataLoader()
pass
