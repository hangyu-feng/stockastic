from alpha_vantage.timeseries import TimeSeries
import json
import pandas

from dataset import raw_to_dataset
from config import DATA_PATH, CREDENTIALS_PATH

class DataLoader:

    def __init__(self, apikey=None) -> None:
        self.apikey = apikey if apikey else self.load_apikey()
        self.ts = TimeSeries(key=self.apikey, output_format='pandas')

    def load_apikey(self, fpath=CREDENTIALS_PATH):
        with open(fpath, 'r') as credentials:
            return json.load(credentials)['alpha_vantage']['api_key']

    def csv_path(self, symbol, interval):
        """
        symbol: (str) stock symbol
        interval: (str) one of 'daily', 'weekly', 'monthly', '1min', '5min', '15min', '30min', '60min'
        """
        return f"{DATA_PATH}/{interval}/{symbol}.csv"

    def save_timeseries_daily(self, symbol):
        """ pull data from alphavantage.co and save to csv file """
        data, metadata = self.ts.get_daily(symbol=symbol, outputsize='full')
        data.to_csv(self.csv_path(symbol, 'daily'))

    def read_timeseries_daily(self, symbol):
        """ read from saved csv file """
        return pandas.read_csv(self.csv_path(symbol, 'daily'))

    def dataset(self, symbol, update=False):
        """ grab data from api and return the processed dataset. If update is False, it will only return
        saved data """
        if update is True:
            self.save_timeseries_daily(symbol)
        raw = self.read_timeseries_daily(symbol)  # will throw error if file not found
        return raw_to_dataset(raw)
