from os.path import dirname

# paths
ROOT_PATH = dirname(dirname(__file__))
CREDENTIALS_PATH = f"{ROOT_PATH}/credentials.json"
DATA_PATH = f"{ROOT_PATH}/data"
MEGA_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404898388(200B+).csv"
LARGE_COMPANIES_PATH = f"{ROOT_PATH}/data/meta/nasdaq_screener_1629404871627(10B+).csv"
MODEL_PATH = f"{ROOT_PATH}/data/models"

# companies
SELECTED_COMPANIES = ['AMD', 'IBM', 'MSFT', 'PCG', 'WMT']

# model compiling
WIN_SIZE = 50
BATCH_SIZE = 32
MODEL_DEFAULT = {
    'lstm_units': 50,
    'drop_rate': 0.2,
    'dense_units': [64, 1],
    'lr': 0.0005,
    'loss': 'mse',
    # 'window_size': WIN_SIZE
}

# model training
SPLIT_RATIO = [0.7, 0.15, 0.15]
FITTING_PARAMS = {
    'epochs': 50,
}
