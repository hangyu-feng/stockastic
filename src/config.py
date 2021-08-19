import pandas
from os.path import dirname

ROOT_PATH = dirname(dirname(__file__))
CREDENTIALS_PATH = f"{ROOT_PATH}/credentials.json"
DATA_PATH = f"{ROOT_PATH}/data"
MEGA_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404898388(200B+).csv"
LARGE_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404871627(10B+).csv"

WIN_SIZE = 50
MODEL_DEFAULT = {
    'lstm_units': 50,
    'drop_rate': 0.2,
    'dense_units': [64, 1],
    'lr': 0.0005,
    'adam_loss': 'mse',
    'window_size': WIN_SIZE
}


MEGA_COMPANIES = list(pandas.read_csv(MEGA_COMPANIES_PATH)['Symbol'])
LARGE_COMPANIES = list(pandas.read_csv(LARGE_COMPANIES_PATH)['Symbol'])
SELECTED_COMPANIES = ['AMD', 'IBM', 'MSFT', 'PCG', 'WMT']
