from functools import wraps
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers

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
def generate_model(lstm_units, drop_rate, dense_units, lr, adam_loss, window_size) -> Model:
    " return a keras model. see https://keras.io/api/models/model_training_apis/ "
    lstm_input = Input(shape=(window_size, 5), name='lstm_input')
    x = LSTM(lstm_units, name='lstm_0')(lstm_input)
    x = Dropout(drop_rate, name='lstm_dropout_0')(x)
    x = Dense(dense_units[0], name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(dense_units[1], name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    mdl = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=lr)
    mdl.compile(optimizer=adam, loss=adam_loss)
    return mdl
