# variables and constants
import sys
sys.path.append("C:/Users/VailG/codes/stockastic/src")
from data_loader import DataLoader
import tensorflow as tf
from window_generator import WindowGenerator, compile_and_fit
from matplotlib import pyplot as plt
import IPython

symbol = 'WMT'

# prepare dataset
loader = DataLoader()
raw = loader.get_raw(symbol)
df = raw[::-1].drop(index=raw.index[0], axis=0)
n = len(df)
train_df = train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]
num_features = df.shape[1]

# data normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')

# windowing
window = WindowGenerator(input_width=5, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df)

# model and fitting
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, window)

# evaluation

IPython.display.clear_output()
val_performance = lstm_model.evaluate(window.val)
performance = lstm_model.evaluate(window.test, verbose=0)

print(val_performance)
print(performance)

pass
