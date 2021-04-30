class SoftmaxReg:
	@staticmethod
	def getBatches(X, y, batch_size = 10):
		sample_size = X.shape[0]

		for i in range(0, sample_size, batch_size):
			batch_X, batch_y = X[i:(i+batch_size), :], y[i:(i+batch_size)]
			yield batch_X, batch_y
	@classmethod
	def gradient_loss(cls, theta, index, X, y):
		number_sample, dimension = X.shape[:2]

		X_theta = X.dot(theta)
		exponent = np.exp(X_theta)
		softmax = exponent/exponent.sum(axis = 1).reshape((number_sample, 1))
		softmax_index = softmax[:, index].reshape((number_sample, 1))

		bool_y = (y == index).astype(np.int).reshape((number_sample, 1))

		gradient = 1/number_sample * ((-bool_y + softmax_index) * X).sum(axis = 0).reshape((dimension, 1))

		return gradient

	def __init__(self, nb_epoch, nb_classes, batch_size = None, learning_rate = 1e-3):
		self.epoch, self.classes, self.batch_size, self.rate = nb_epoch, nb_classes, batch_size, learning_rate
			self.gradient = lambda theta, index, batch_X, batch_y: (
			SoftmaxReg.gradient_loss(theta, index, batch_X, batch_y)
		)
		self.theta = None

	def fit(self, X, y, init_theta = None, random_state = 0):
		rnd = np.random.RandomState(random_state)
		X = X.copy()
		y = y.reshape((y.shape[0], 1))
		sample_size, dimension = X.shape[:2]

		if init_theta is None:
			self.theta = (
				rnd.normal(loc = 0, scale = 1, size = dimension * self.classes)
					.reshape((dimension, self.classes))
			)
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
				for j in range(self.classes):
					col_theta = self.theta[:, j].reshape((dimension, 1))
					col_theta = col_theta - self.rate * self.gradient(self.theta, j, batch_X, batch_y)
					self.theta[:, j] = col_theta.flatten()
		return self

	def predict(self, X):
		assert self.theta is not None, 'Model needs to fit to a training set before making prediction'

		if len(X.shape) == 1: # Predict one sample
			dimension, = X.shape
			X = X.reshape((1, dimension))

		number_sample, dimension = X.shape[:2]
		X_theta = X.dot(self.theta)
		exponent = np.exp(X_theta)
		softmax = exponent/exponent.sum(axis = 1).reshape((number_sample, 1))
		predict_y = softmax.argmax(axis = 1)

		return predict_y
