class NaiveBayesClassifier:

	def __init__(self, laplace = True):
		self.laplace = laplace

	def fit(self, X, y, categorical_features = None, numerical_features = None, labels = (0, 1)):
		self.label_dict = {i : y[y == i].shape[0]/y.shape[0] for i in labels}
		self.unique_val = dict()
		self.prob_dict = dict()

		if categorical_features is None:
			categorical_prob = dict()
		else:
			categorical_prob = dict()
			for i in categorical_features:
				categorical_prob.setdefault(i, dict())
				col_i_prob = dict()
				col_i = X[:, i]
				self.unique_val.setdefault(i, 0)
				self.unique_val[i] = np.unique(col_i).shape[0]
				for x in np.unique(col_i):
					col_i_prob.setdefault(x, dict())
					if self.laplace:
						col_i_prob[x] = {
							j : (np.sum((col_i == x) & (y == j)) + 1)/(np.sum(y == j) + self.unique_val[i])\
							for j in labels
						}
					else:
						col_i_prob[x] = {
							j : np.sum((col_i == x) & (y == j))/np.sum(y == j) for j in labels
						}
				categorical_prob[i] = col_i_prob

		if numerical_features is None:
			numerical_prob = dict()
		else:
			numerical_prob = dict()
			for i in numerical_features:
				col_i = X[:, i]
				numerical_prob.setdefault(i, dict())
				for j in labels:
					mean, std = (col_i[y == j]).mean(), (col_i[y == j]).std(ddof = 1)
					numerical_prob[i][j] = (mean, std)

		self.prob_dict.update(categorical_prob)
		self.prob_dict.update(numerical_prob)

		self.numerical = numerical_features
		self.categorical = categorical_features
		self.labels = labels

		return self
	def predict(self, X):
		y_predict = []

		for x in X:
			predict_value = None
			prob_max = -1
			for label in self.labels:
				probs = []

				if self.categorical is not None:
					for categorical_col in self.categorical:
						if categorical_col in self.prob_dict.keys():
							if x[categorical_col] in self.prob_dict[categorical_col].keys():
								probs.append(self.prob_dict[categorical_col][x[categorical_col]][label])
							else:
								probs.append(1/self.unique_val[categorical_col])
						else:
								probs.append(1/self.unique_val[categorical_col])                            

				if self.numerical is not None:
					for numerical_col in self.numerical:
						mean, std = self.prob_dict[numerical_col][label]
						density = 1/(std * sqrt(2*pi)) * exp**(-(x[numerical_col] - mean)**2/(2 * std**2))
						probs.append(density)

				prob_label = self.label_dict[label] * np.cumprod(probs)[-1]
				if prob_label > prob_max:
					predict_value, prob_max = label, prob_label
			y_predict.append(predict_value)
		return np.array(y_predict)
