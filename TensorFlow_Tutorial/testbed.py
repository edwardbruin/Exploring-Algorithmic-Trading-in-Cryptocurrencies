import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def json_to_csv(filename):
	testjson = pd.read_json(filename)
	filename = filename.removesuffix('.json')+'.csv'
	testjson.to_csv(filename,index=False, header=False)

def read_csv(filename):
	csv_data = pd.read_csv(filename)
	return csv_data
	
def plot_acc(kl, model, kf):
	pred = model(kf)
	plt.figure()
	plt.plot(kl, label='actual')
	plt.plot(pred, label='predicted')
	plt.legend()
	plt.show()

def regular_model(kline_features, kline_labels):
	layer1_neurons = int(input("layer1_neurons: "))
	epochs = int(input("epochs: "))
	kline_model = tf.keras.Sequential([
		layers.Dense(layer1_neurons),
		layers.Dense(1)
	])
	kline_model.compile(loss = tf.losses.MeanSquaredError(),
						optimizer = tf.optimizers.Adam())
	kline_model.fit(kline_features, kline_labels, epochs=epochs, verbose=0)
	kline_model.evaluate(kline_features, kline_labels, verbose=2)
	return kline_model

def normalised_model(kline_features, kline_labels):
	layer1_neurons = int(input("layer1_neurons: "))
	epochs = int(input("epochs: "))
	normalize = preprocessing.Normalization()
	normalize.adapt(kline_features)
	normalizel = preprocessing.Normalization()
	normalizel.adapt(kline_labels)		
	norm_kline_model = tf.keras.Sequential([
		normalize,
		normalizel,
		layers.Dense(layer1_neurons),
		layers.Dense(4)
	])
	norm_kline_model.compile(	loss = tf.losses.MeanSquaredError(),
							  	optimizer = tf.optimizers.Adam())
	norm_kline_model.fit(kline_features, kline_labels, epochs=epochs, verbose=0)
	norm_kline_model.evaluate(kline_features, kline_labels, verbose=2)
	return norm_kline_model

# def process_inputs(norm):	
def process_inputs():	
	header_names = ["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time", "Quote_asset_volume", "Number_of_trades", "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"]
	usecols = header_names[1:6] + header_names[7:11]
	kline_train = pd.read_csv("klinedata.csv", names = header_names, usecols = usecols)
	training_target = header_names[8]
	json_to_csv('klinedata.json')
	kline_train.head()

	kline_features = kline_train.copy()
	#kline_labels = kline_features.pop(training_target)
	kline_labels = kline_features.Close[1:]
	kline_features.drop(kline_features.tail(1).index, inplace=True)
	kline_features = np.array(kline_features)
	return kline_features, kline_labels

class SMA_List:
	def __init__(self, long, short):
		self.long = long
		self.short = short
	long = []
	short = []

def create_SMA_file(payload1, filename):
	df = pd.DataFrame(data = [payload1.long, payload1.short])
	df.to_csv(filename+'.csv',index=False, header=False)

def SMAs(kt, long_length, short_length):
	test = SMA_List
	listLen = len(kt)
	foobar = pd.DataFrame()
	i = long_length
	j = short_length
	longSMA = np.zeros(listLen)
	shortSMA = np.zeros(listLen)
	for n in range(listLen):
		bot = 0 if n-i < 0 else n-i
		top = listLen if n+i+1 > listLen else n+i+1
		kt.Close[bot:top]
		kt.Close[bot:top].mean()
		longSMA[n] = kt.Close[bot:top].mean()
		bot = 0 if n-j < 0 else n-j
		top = listLen if n+j+1 > listLen else n+j+1
		shortSMA[n] = kt.Close[bot:top].mean()
	longSMA[:i] = 0
	longSMA[-i:] = 0
	shortSMA[:j] = 0
	shortSMA[-j:] = 0
	test.long = longSMA
	test.short = shortSMA
	return test

def create_target_file(payload, filename):
	out = calculate_targets(payload['Close'])
	df = pd.DataFrame(data = out)
	df.to_csv(filename+'.csv',index=False, header=False)

def calculate_median(input_candle):
	total = input_candle[1] + input_candle[2]
	average = total / 2
	return average
	
# will calculate the difference between two data points. if given 
# more than two, will calculate between the last two in the list
def calculate_difference(close_values):
	arrayLength = len(close_values)-1
	if (arrayLength>0):
		difference = close_values[arrayLength] - close_values[arrayLength-1]
	return difference

# will calculate a list of targets for the training function to use
def calculate_targets(Close_values):
	targets = []
	length = len(Close_values) - 2
	if (length > 0):
		for foo in range(1, length+1):
			difference = calculate_difference([Close_values[foo],Close_values[foo+1]])
			targets.append(difference)
	return targets
