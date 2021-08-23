from functools import wraps
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import metrics
from tensorflow.keras import optimizers

from config import MODEL_DEFAULT

def default_args(defaults):
    """ return a decorator that takes a function as input """
    def wrapper(func):
        @wraps(func)  # just to show docstring of original function
        def new_func(*args, **kwargs):
            kwargs = defaults | kwargs
            return func(*args, **kwargs)
        return new_func
    return wrapper

@default_args(MODEL_DEFAULT)
def generate_model(lstm_units, drop_rate, dense_units, lr, loss):
    """ return a keras model. see https://keras.io/api/models/model_training_apis/
    all parameters are optional. default parameters are in config.py
    """
    lstm_model = Sequential([
        LSTM(lstm_units, return_sequences=True),
        Dropout(drop_rate),
        Dense(dense_units[0], activation='sigmoid'),
        Dense(dense_units[1])
    ])
    adam = optimizers.Adam(learning_rate=lr)
    lstm_model.compile(optimizer=adam, loss=loss)
    return lstm_model
