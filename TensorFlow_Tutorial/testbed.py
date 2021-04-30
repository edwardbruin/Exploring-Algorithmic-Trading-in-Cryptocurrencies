import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BNN:
	def __init__(self):
		self.filename = 'klinedata.json'
		self.trainfilename = 'klinedata.json'
		self.testfilename = 'test.json'
		self.kl = None
		self.kf = None
		self.ktest = None
		self.ktestlabels = None
		self.model = None
		self.offset = 1
		self.neurons = 128
		self.epochs = 15
	
	def set_trainfilename(self):
		new_filename = input("enter new training json filename: ")
		self.trainfilename = new_filename
	
	def set_testfilename(self):
		new_filename = input("enter new testing json filename: ")
		self.testfilename = new_filename
	
	def set_offset(self):
		new_offset = int(input("enter new offset: "))
		self.offset = new_offset
	
	def set_neurons(self):
		new_neurons = int(input("enter new neurons: "))
		self.neurons = new_neurons
	
	def set_epochs(self):
		new_epochs = int(input("enter new epochs: "))
		self.epochs = new_epochs
	
	def set_invalid(self):
		print("not a valid selection")
	
	def set_variables(self):
		selection = 0
		print("1-trainfilename")
		print("2-offset")
		print("3-neurons")
		print("4-epochs")
		print("5-testfilename")
		selection = int( input("enter selection: ") )
		switcher = {
			1: self.set_trainfilename,
			2: self.set_offset,
			3: self.set_neurons,
			4: self.set_epochs,
			5: self.set_testfilename
		}
		switcher.get( selection, self.set_invalid )()

	def json_to_csv(self):
		filename = self.trainfilename
		testjson = pd.read_json(filename)
		filename = filename.removesuffix('.json')+'.csv'
		testjson.to_csv(filename,index=False, header=False)

	def read_csv(filename):
		csv_data = pd.read_csv(filename)
		return csv_data
		
	def evaluate_model(self):
		foo = self.model.evaluate(self.ktest, self.ktestlabels, verbose=2)
		return foo
		
	def plot_acc(self):
		kl = self.ktestlabels
		model = self.model
		kf = self.ktest
		pred = model(kf)
		plt.figure()
		plt.grid()
		plt.plot(kl, label='actual')
		plt.plot(pred, label='predicted')
		plt.legend()
		plt.show()

	def regular_model(self):
		kline_features = self.kf
		kline_labels = self.kl
		layer1_neurons = self.neurons
		epochs = self.epochs
		kline_model = tf.keras.Sequential([
			layers.Dense(layer1_neurons),
			layers.Dense(1)
		])
		kline_model.compile(loss = tf.losses.MeanSquaredError(),
							optimizer = tf.optimizers.Adam())
		kline_model.fit(kline_features, kline_labels, epochs=epochs, verbose=0)
		#kline_model.evaluate(kline_features, kline_labels, verbose=2)
		self.model = kline_model
		# return kline_model

	def normalised_model(self):
		kline_features = self.kf
		kline_labels = self.kl
		layer1_neurons = self.neurons
		epochs = self.epochs
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
		#norm_kline_model.evaluate(kline_features, kline_labels, verbose=2)
		# return norm_kline_model

	# def process_inputs(norm):	
	def process_inputs(self):	
		offset = self.offset
		header_names = ["Open_time", "Open", "High", "Low", "Close", "Volume", "Close_time", "Quote_asset_volume", "Number_of_trades", "Taker_buy_base_asset_volume", "Taker_buy_quote_asset_volume", "Ignore"]
		usecols = header_names[1:6] + header_names[7:11]
		kline_train = pd.read_csv("klinedata.csv", names = header_names, usecols = usecols)
		training_target = header_names[8]
		
		kline_test = pd.read_csv("klinedata.csv", names = header_names, usecols = usecols)
		kline_test_labels = kline_test.Close[offset:]
		kline_test.drop(kline_test.tail(offset).index, inplace=True)
		
		kline_features = kline_train.copy()
		kline_labels = kline_features.Close[offset:]
		kline_features.drop(kline_features.tail(offset).index, inplace=True)
		self.kf = np.array(kline_features)
		self.kl = np.array(kline_labels)
		self.ktest = np.array(kline_test)
		self.ktestlabels = np.array(kline_test_labels)
		# return self.kf, self.kl

	class SMA_List:
		def __init__(self, long, short):
			self.long = long
			self.short = short
		long = []
		short = []
		
	def append_SMA(kf, filename):
		SMAs = pd.read_csv(filename, names=['long', 'short'])

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
		return longSMA, shortSMA

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
	
	def loop(self):
		self.regular_model()
		return self.evaluate_model()
		
def optimize(offset):
	BNN1 = BNN()
	BNN2 = BNN()
	
	BNN1.offset = offset
	BNN2.offset = offset
	
	BNN1.process_inputs()
	BNN2.process_inputs()
	
	BNN1.regular_model()
	BNN2.regular_model()
	
	for i in range(10):
		acc1 = BNN1.evaluate_model()
		acc2 = BNN2.evaluate_model()
		if (acc1 > acc2):
			BNN1.regular_model()
		
		if (acc2 > acc1):
			BNN2.regular_model()
	winner = [acc1, acc2].index(min([acc1, acc2]))
	print(winner)
	winnerdict = {
		0:BNN1,
		1:BNN2
	}
	return winnerdict.get(winner)