from alpha_vantage.timeseries import TimeSeries
import json


def get_apikey():
    with open("credentials.json", 'r') as credentials:
        return json.load(credentials)['alpha_vantage']['api_key']


class DataLoader:

    def __init__(self, apikey=None) -> None:
        self.apikey = apikey if apikey else self.load_apikey()

    def load_apikey(self, fpath="credentials.json"):
        with open(fpath, 'r') as credentials:
            return json.load(credentials)['alpha_vantage']['api_key']

    def save_timeseries_daily(self, symbol):
        ts = TimeSeries(key=self.apikey, output_format='csv')
        output = ts.get_daily(symbol=symbol, outputsize='full')
        pass
