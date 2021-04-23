import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import matplotlib as plt

# layer1_neurons = 64
# epochs = 20
layer1_neurons = int(input("layer1_neurons"))
epochs = int(input("epochs"))
header_names = ["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time", "Quote_asset_volume", "Number_of_trades", "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"]
usecols = header_names[1:6] + header_names[7:11]
training_target = header_names[8]

kline_train = pd.read_csv("klinedata.csv", names = header_names, usecols = usecols)
kline_train.head()

kline_features = kline_train.copy()
kline_labels = kline_features.pop(training_target)
kline_features = np.array(kline_features)
print(f"data loaded")
input()
kline_model = tf.keras.Sequential([
	layers.Dense(layer1_neurons),
	layers.Dense(1)
])
print(f"model defined")
input()

kline_model.compile(loss = tf.losses.MeanSquaredError(),
					optimizer = tf.optimizers.Adam())
print(f"model compiled")
input()
					
kline_model.fit(kline_features, kline_labels, epochs=epochs)
print(f"model fitted")
input()
kline_model.evaluate(kline_features, kline_labels, verbose=2)
print(f"evaluated")
input()