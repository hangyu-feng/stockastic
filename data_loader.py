from alpha_vantage.timeseries import TimeSeries
import json
import pandas
class DataLoader:

    def __init__(self, apikey=None) -> None:
        self.apikey = apikey if apikey else self.load_apikey()
        self.ts = TimeSeries(key=self.apikey, output_format='pandas')

    def load_apikey(self, fpath="credentials.json"):
        with open(fpath, 'r') as credentials:
            return json.load(credentials)['alpha_vantage']['api_key']

    def csv_path(self, symbol, interval):
        """
        symbol: (str) stock symbol
        interval: (str) one of 'daily', 'weekly', 'monthly', '1min', '5min', '15min', '30min', '60min'
        """
        return f"./data/{interval}/{symbol}.csv"

    def save_timeseries_daily(self, symbol):
        """ pull data from alphavantage.co and save to csv file """
        data, metadata = self.ts.get_daily(symbol=symbol, outputsize='full')
        data.to_csv(self.csv_path(symbol, 'daily'))

    def read_timeseries_daily(self, symbol):
        """ read from saved csv file """
        return pandas.read_csv(self.csv_path(symbol, 'daily'))
