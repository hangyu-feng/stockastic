from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow import metrics
from tensorflow.keras import optimizers

from config import MODEL_DEFAULT
from default_args import default_args

@default_args(MODEL_DEFAULT)
def generate_model(lstm_units, drop_rate, dense_units, lr, loss):
    """ return a keras model. see https://keras.io/api/models/model_training_apis/
    all parameters are optional. default parameters are in config.py
    """
    lstm_model = Sequential([
        LSTM(lstm_units, return_sequences=True),
        Dropout(drop_rate),
        Dense(dense_units[0], activation='relu'),
        Dense(dense_units[1])
    ])
    adam = optimizers.Adam(learning_rate=lr)
    lstm_model.compile(optimizer=adam, loss=loss)
    return lstm_model
