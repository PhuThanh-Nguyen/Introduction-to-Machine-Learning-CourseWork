class LinearReg:
	@staticmethod
	def getBatches(X, y, batch_size = 10):
		sample_size = X.shape[0]

		for i in range(0, sample_size, batch_size):
			batch_X, batch_y = X[i:(i+batch_size), :], y[i:(i+batch_size)]
			yield batch_X, batch_y

	@classmethod
	def MSE(cls, theta, X, y):
		y_predict = X.dot(theta)
		number_sample = X.shape[0]
		loss = (1/(2 * number_sample) * (y_predict - y).T.dot(y_predict - y)).item()
		return loss

	def __init__(self, nb_epoch, batch_size = None, learning_rate = 1e-3):
	    '''
			LinearReg's constructor
			------------------------------
			Parameters:
				nb_epoch: int
					Number of epoches
				batch_size: int, default None
					If batch_size is None then perform Batch Gradient Descent
					If batch_size == 1 then perform Stochastic Gradient Descent
					If batch_size > 1 then perform Mini Batch Gradient Descent
				learning_rate: float, default 1e-3
		'''
		self.epoch, self.batch_size, self.rate = nb_epoch, batch_size, learning_rate

		# Define MSE loss function and its gradient

		self.loss_function = lambda theta, batch_X, batch_y: (
			1/(2 * batch_X.shape[0]) * (batch_X.dot(theta) - batch_y).T.dot(batch_X.dot(theta) - batch_y)
		).item()
        
		self.gradient = lambda theta, batch_X, batch_y: 1/(batch_X.shape[0]) * (
			batch_X.T.dot(batch_X.dot(theta) - batch_y)
		)

		self.theta = None

	def fit(self, X, y, init_theta = None, random_state = 0):
		'''
			Fit linear model
			----------------------------
			Parameters:
				X: np.ndarray of shape (sample_size, dimension)
					Training data
				y: np.ndarray of shape (sample_size, 1)
					Target values
				init_theta: np.ndarray of shape (dimension, 1), default None
					Initial value for theta
					If None, initial value for theta will be chosen by normal distribution N(0, 1)
				random_stae: int, default 0
					Random state to set initial theta and to shuffle data for each epoch
			----------------------------
			Returns: np.ndarray
				MSE loss history for each update
		'''
		rnd = np.random.RandomState(random_state)
		X = X.copy()
		y = y.reshape((y.shape[0], 1))
		sample_size, dimension = X.shape[:2]

		if init_theta is None:
			self.theta = rnd.normal(loc = 0, scale = 1, size = dimension).reshape((dimension, 1))
		else:
			self.theta = init_theta.copy()

		# If it is BGD (batch_size = None) then not shuffle else shuffle dataset
		shuffle = True
		if self.batch_size == None:
			self.batch_size = sample_size
			shuffle = False
        
		# Save MSE loss for each update
		loss_history = []

		for i in range(self.epoch):
			if shuffle:
				# Stack X and y horizontally
				data = np.hstack((X, y))
				# Shuffle inplace
				rnd.shuffle(data)
				# Get back X, y after shuffle
				X, y = data[:, :dimension], data[:, dimension:]

			for batch_X, batch_y in LinearReg.getBatches(X, y, batch_size = self.batch_size):
				# Update theta
				self.theta = self.theta - self.rate * self.gradient(self.theta, batch_X, batch_y)
                
				loss = self.loss_function(self.theta, batch_X, batch_y)
				loss_history.append(loss)

		return np.array(loss_history)

	def predict(self, X):
		'''
			Predict using the linear model.
			---------------------
			Parameters:
				X: np.ndarray
					Samples
			---------------------
			Returns: np.ndarray
		'''

		assert self.theta is not None, 'Model needs to fit to a training set before making prediction'

		if len(X.shape) == 1: # Predict one sample
			dimension, = X.shape
			X = X.reshape((1, dimension))
		predict_y = X.dot(self.theta)

		return predict_y
