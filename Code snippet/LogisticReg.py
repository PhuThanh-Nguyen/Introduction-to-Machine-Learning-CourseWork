class LogisticReg:
	@staticmethod
	def getBatches(X, y, batch_size = 10):
		sample_size = X.shape[0]

		for i in range(0, sample_size, batch_size):
			batch_X, batch_y = X[i:(i+batch_size), :], y[i:(i+batch_size)]
			yield batch_X, batch_y

	@classmethod
	def gradient_loss(cls, theta, X, y, lmbda = 0):
		h_theta = 1/(1 + np.exp(-X.dot(theta)))
		number_sample, dimension = X.shape[:2]

		regularization_term = np.zeros((dimension, 1))

		regularization_term[1:] = 2 * theta[1:]

		gradient = 1/number_sample * X.T.dot(h_theta - y) + lmbda * regularization_term

		return gradient

	def __init__(self, nb_epoch, batch_size = None, learning_rate = 1e-3, lmbda = 0):
		'''
			LogisticReg's constructor
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

		# Define loss's gradient

		self.gradient = lambda theta, batch_X, batch_y: LogisticReg.gradient_loss(theta, batch_X, batch_y, lmbda)

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
				random_state: int, default 0
					Random state to set initial theta and to shuffle data for each epoch
			----------------------------
			Returns: LogisticReg's instance
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

		for i in range(self.epoch):
			if shuffle:
				# Stack X and y horizontally
				data = np.hstack((X, y))
				# Shuffle inplace
				rnd.shuffle(data)
				# Get back X, y after shuffle
				X, y = data[:, :dimension], data[:, dimension:]

			for batch_X, batch_y in LogisticReg.getBatches(X, y, batch_size = self.batch_size):
				# Update theta
				self.theta = self.theta - self.rate * self.gradient(self.theta, batch_X, batch_y)

		return self

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
		predict_y = np.where(X.dot(self.theta) < 0, 0, 1)

		return predict_y
