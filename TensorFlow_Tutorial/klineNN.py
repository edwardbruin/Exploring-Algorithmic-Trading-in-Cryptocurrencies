import numpy as np

class NeuralNetwork:
	def __init__(self, learning_rate):
		self.weights = np.array([np.random.randn(), np.random.randn(), np.random.randn(), np.random.randn()])
		self.weights_row = np.array([self.weights, self.weights, self.weights])
		self.weight_sets = np.array([self.weights_row, self.weights_row, self.weights_row])
		self.volume_threshold = 1.05
		self.SMA_threshold = 1.05
		self.SMA_length = 10
		self.accuracy = np.zeros((3,3))
		self.innacuracy = np.zeros((3,3))
		self.bias = np.random.randn()
		self.learning_rate = learning_rate
		
	def _sigmoid(self, x):
		return 1 / (1 + np.exp(-x))
		
	def _sigmoid_deriv(self, x):
		return self._sigmoid(x) * (1 - self._sigmoid(x))
	
	def predict(self, input_vector):
		layer_1 = np.dot(input_vector, self.weights) + self.bias
		layer_2 = self._sigmoid(layer_1)
		prediction = layer_2
		return prediction
		
	def _compute_gradients(self, input_vector, target):
		layer_1 = np.dot(input_vector, self.weights) + self.bias
		layer_2 = self._sigmoid(layer_1)
		prediction = layer_2
		
		derror_dprediction = 2 * (prediction - target)
		dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
		dlayer1_dbias = 1
		dlayer1_dweights = (0 * self.weights) + (1 * input_vector)
		
		derror_dbias = (
			derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
		)
		derror_dweights = (
			derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
		)
		
		return derror_dbias, derror_dweights
		
	def _update_parameters(self, derror_dbias, derror_dweights):
		self.bias = self.bias - (derror_dbias * self.learning_rate)
		self.weights = self.weights - (
			derror_dweights * self.learning_rate
		)
	
	# will calculate the Simple Moving Average of the given data
	def calculate_SMA(input_candles):
		length = len(input_candles)
		SMA_length = self.SMA_length
		
		startPoint = 0
		if (length - SMA_length > 0):
			startPoint = length - SMA_length
			
		total = 0
		for foo in range(startPoint, length):
			currentmedian = calculate_median(input_candles[foo])
			total = total + currentmedian
		average = total / SMA_length
		return average
	
	# will calculate a 'median' actually just the average between
	# the opening and closing values of the candlestick
	def calculate_median(input_candle):
		total = input_candle[1] + input_candle[2]
		average = total / 2
		return average
		
	# will calculate the difference between two data points. if given 
	# more than two, will calculate between the last two in the list
	def calculate_difference(input_candles):
		arrayLength = len(input_candles)-1
		difference = [0,0,0,0]
		if (arrayLength>0):
			for foo in range(0, len(input_candles[0])-1):
				difference[foo] = input_candles[arrayLength][foo] - input_candles[arrayLength-1][foo]
		return difference
	
	# will calculate a list of targets for the training function to use
	def calculate_targets(input_candles):
		targets = []
		length = len(input_candles) - 2
		if (length > 0):
			for foo in range(1, length+1):
				difference = calculate_difference([input_candles[foo],input_candles[foo+1]])
				median = calculate_median(difference)
				targets.append(median)
		return targets
		
	def read_json(self, filename):
		data = []
		dataFloat = []
		with open(filename) as f:
			data = json.load(f)
		for i in data:
			dataFloat.append([float(j) for j in i])
		return dataFloat
		
	def write_json(data, filename):
		with open(filename, 'w') as outfile:
			json.dump(data, outfile)
		
	def transform(vecs):
		length = len(vecs)-1
		newVecs = []
		for foo in range(0, length+1):
			newVecs.append([float(vecs[foo][1]), float(vecs[foo][2]), float(vecs[foo][3]), float(vecs[foo][4]),  float(vecs[foo][5])])
		return newVecs
		
	# will analyse to see if the new volume has gone above or below
	# a given boundary when compared to the previous volume
	def detect_volume_change(vecs):
		volume_threshold = self.volume_threshold
		length = len(vecs)-1
		volume1 = vecs[length-1][4]
		volume2 = vecs[length][4]
		if (volume1/volume2>volume_threshold):
			#below volume
			return -1
		elif (volume2/volume1>volume_threshold):
			#above volume
			return 1
		else:
			#within the average threshold
			return 0
		
	# will detect if the new price has gone above or below the 
	# Simple Moving Average, within a certain boundary
	def detect_price_change(input_candle, vecs):
		SMA_threshold = self.SMA_threshold
		SMA = calculate_SMA(vecs)
		median = calculate_median(input_candle)
		if (SMA/median>SMA_threshold):
			#below SMA
			return -1
		elif (median/SMA>SMA_threshold):
			#above SMA
			return 1
		else:
			#within the average threshold
			return 0
		
	def train(self, input_candles, targets, iterations):
		cumulative_errors = []
		for current_iteration in range(iterations):
			#pick a data instance at random, except for the final data instance
			random_data_index = np.random.randint(len(input_candles)-1)
			
			difference = calculate_difference(input_candles[random_data_index:random_data_index+1])
			target = targets[random_data_index]
			
			#Compute the gradients and update the weights
			derror_dbias, derror_dweights = self._compute_gradients(
				difference, target
			)
			
			self._update_parameters(derror_dbias, derror_dweights)
			
			#Measure the cumulative error for all the instances
			if current_iteration % 100 == 0:
				cumulative_error = 0
				#loop through all the instances to measure the error
				for data_instance_index in range(len(input_candles)):
					data_point = input_candles[data_instance_index]
					target = targets[data_instance_index]
					
					prediction = self.predict(data_point)
					error = np.square(prediction - target)
					
					cumulative_error = cumulative_error + error
				cumulative_errors.append(cumulative_error)
				
		return cumulative_errors