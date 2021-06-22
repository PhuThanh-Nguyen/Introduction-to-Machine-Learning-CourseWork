import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class InputLayer:
	def __init__(self, inputs):
		self.inputs = inputs.copy()
		self.ac_X = self.inputs

class HiddenLayer:
	def __init__(self, activation_func, derivative_activation, input_shape, output_shape, weights = None, bias = None, random_state = 0):
		self.activation, self.derivative = activation_func, derivative_activation
		self.inp, self.outp = input_shape, output_shape
		if weights is None:
			rnd = np.random.RandomState(random_state)
			self.weights = rnd.normal(loc = 0, scale = 1, size = (self.outp, self.inp))
		else:
			self.weights = weights.copy()

		if bias is None:
			rnd = np.random.RandomState(random_state)
			self.bias = rnd.normal(loc = 0, scale = 1, size = (self.outp, 1))
		else:
			self.bias = bias.copy()

	def forward(self, X):
		assert X.shape[0] == self.inp, f'{X.shape} and {self.weights.shape} do not match'
		self.z_X = self.weights @ X + self.bias
		self.ac_X = self.activation(self.z_X)
		return self.ac_X

	def update(self, weight_derivative, bias_derivative, lr = 1e-3):
		self.weights -= lr * weight_derivative
		self.bias -= lr * bias_derivative

# Implementing delta rule based on: http://cs229.stanford.edu/notes-spring2019/backprop.pdf
class NeuralNetwork:

	def __init__(self, layers):

		assert isinstance(layers[0], InputLayer), 'First layer in layers list must be an instance of InputLayer'
		for i, layer in enumerate(layers[1:]):
			assert isinstance(layer, HiddenLayer), f'{i + 1}th layer is not an instance of HiddenLayer'
		self.layers, self.number_layers = layers, len(layers)

	def forward_fit(self):
		return self.forward(self.layers[0].inputs)

	def forward(self, X):
		current_X = X
		for layer in self.layers[1:]:
			current_X = layer.forward(current_X)
		outputs = current_X.copy()
		return outputs

	def __backpropagation(self, deltas, lr = 1e-3):

		assert len(deltas) == 1 and isinstance(deltas, list)

		for l in range(self.number_layers - 2, 0, -1):
			delta_l = (self.layers[l + 1].weights.T @ deltas[-1]) * self.layers[l].derivative(self.layers[l].z_X)
			deltas.append(delta_l)

		deltas = [None,] + deltas[::-1]

		for l in range(1, self.number_layers):
			weights_gradient = deltas[l] @ self.layers[l - 1].ac_X.T
			bias_gradient = np.sum(deltas[l], axis = 1).reshape(-1, 1)
			self.layers[l].update(weights_gradient, bias_gradient, lr)

	def backward(self, lr, deltas = None):
		self.__backpropagation(deltas, lr = lr)
		return self

def one_hot_vector(y):
	out = np.zeros((y.shape[0], y.max() + 1))
	for i in range(y.shape[0]):
		out[i, y[i]] = 1
	return out

def linearFunc(X):
	return X.copy()

def linearDeriv(X):
	return np.ones(X.shape, dtype = X.dtype)

def relu(X):
	return np.where(X >= 0, X, 0).reshape(X.shape)
def relu_deriv(X):
	return np.where(X >= 0, 1, 0).reshape(X.shape)

def softmax(X):
	return np.exp(X)/np.sum(np.exp(X), axis = 0)

def softmax_deriv(X):
	return softmax(X) * (1 - softmax(X))

def crossEntropyLoss(y_pred, y):
	return -np.mean(y * np.log(y_pred))

# Example
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_mean = np.mean(X_train, axis = 0).reshape(1, -1)
X_std = np.std(X_train, axis = 0).reshape(1, -1)

X_valid = (X_valid - X_mean)/X_std
X_train = (X_train - X_mean)/X_std

X_train, X_valid = X_train.T, X_valid.T
y_train = one_hot_vector(y_train).T

# Xavier Weights Initialization for linear/sigmoid activation function
rnd = np.random.RandomState(0)
W_1 = rnd.uniform(low = -1/np.sqrt(30), high = 1/np.sqrt(30), size = 600).reshape(20, 30)
W_2 = rnd.uniform(low = -1/np.sqrt(20), high = 1/np.sqrt(20), size = 400).reshape(20, 20)
W_3 = rnd.uniform(low = -1/np.sqrt(20), high = 1/np.sqrt(20), size = 200).reshape(10, 20)
W_4 = rnd.uniform(low = -1/np.sqrt(10), high = 1/np.sqrt(10), size = 20).reshape(2, 10)

inputs = InputLayer(X_train)

hidden_layer_1 = HiddenLayer(
	linearFunc, linearDeriv, 
	input_shape = 30, output_shape = 20, 
	weights = W_1,
	bias = np.zeros((20, 1))
)

hidden_layer_2 = HiddenLayer(
	softmax, softmax_deriv, 
	input_shape = 20, output_shape = 20,
	weights = W_2,
	bias = np.zeros((20, 1))
)

hidden_layer_3 = HiddenLayer(
	linearFunc, linearDeriv,
	input_shape = 20, output_shape = 10,
	weights = W_3,
	bias = np.zeros((10, 1))
)

outputs_layer = HiddenLayer(
	softmax, softmax_deriv, 
	input_shape = 10, output_shape = 2,
	weights = W_4,
	bias = np.zeros((2, 1))
)

layers = [
	inputs, 
	hidden_layer_1, 
	hidden_layer_2, 
	hidden_layer_3, 
	outputs_layer
]

model = NeuralNetwork(layers)

learning_rate = 1e-5
for epoch in range(10000):
	y_predict = model.forward_fit()
	err = crossEntropyLoss(y_predict, y_train)
	deltas = [y_predict - y_train, ]
	model.backward(learning_rate, deltas = deltas)

	if epoch % 1000 == 0:
		print(f'Loss at epoch {epoch + 1} = {err}')
# Training accuracy
print(np.sum(np.argmax(model.forward(X_train), axis = 0) == np.argmax(y_train, axis = 0))/y_train.shape[1])
