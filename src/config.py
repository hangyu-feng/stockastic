from src.main import TRAIN_SPLIT
import pandas
from os.path import dirname

# paths
ROOT_PATH = dirname(dirname(__file__))
CREDENTIALS_PATH = f"{ROOT_PATH}/credentials.json"
DATA_PATH = f"{ROOT_PATH}/data"
MEGA_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404898388(200B+).csv"
LARGE_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404871627(10B+).csv"

# companies
# MEGA_COMPANIES = list(pandas.read_csv(MEGA_COMPANIES_PATH)['Symbol'])
# LARGE_COMPANIES = list(pandas.read_csv(LARGE_COMPANIES_PATH)['Symbol'])
SELECTED_COMPANIES = ['AMD', 'IBM', 'MSFT', 'PCG', 'WMT']

# model compiling
WIN_SIZE = 50
MODEL_DEFAULT = {
    'lstm_units': 50,
    'drop_rate': 0.2,
    'dense_units': [64, 1],
    'lr': 0.0005,
    'adam_loss': 'mse',
    'window_size': WIN_SIZE
}

# model training
TRAIN_SPLIT = 0.9
FITTING_PARAMS = {
    'batch_size': 32,
    'epochs': 50,
    'shuffle': True,
    'validation_split': 0.1
}
