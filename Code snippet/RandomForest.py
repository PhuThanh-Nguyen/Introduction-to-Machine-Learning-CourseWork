from collections import Counter
import numpy as np
from sklearn.tree import DecisionTreeClassifier

class RandomForestClassifier:
	def __init__(self, number_features, number_trees, random_state = 0):
		'''
			Initialize RandomForestClassifier object
			Parameters:
				number_features: int
					Number of features to use for each decision tree
				number_trees: int
					Number of trees in the forest
				random_state: int, default 0
					Initialize random state to randomly choosing a subset of training data for each tree 
		'''
		self.rnd = np.random.RandomState(random_state)
		self.nb_features, self.nb_trees = number_features, number_trees
		self.trees = None

	def forestVoting_(self, votes):
		'''
			Make `a` prediction from the votes of every trees in the forest
			Parameters:
				votes: list-like object
					Votes from each tree
			Returns:
				A prediction is made from votes, which is based on 'majority votes' scheme
				(A type that function returns depends on type of each element of training label y)
		'''
		count = Counter(votes)
		# Get item with major votes
		result, _ = count.most_common(1)[0]
		return result

	def fit(self, X, y, **kwargs):
		'''
			Fit a training set to classifier.
			Parameters:
				X: numpy.ndarray of shape (n_samples, n_features)
					Training data
				y: numpy.ndarray of shape (n_samples,)
					Training labels corresponding to each sample from training data
				kwargs: Keyword arguments to pass in each Decision Tree
					List of keyword arguments can be found in scikitlearn Decision Tree's documentation
			Returns: RandomForestClassifier object
					RandomForestClassifier object after fitting it to the training set
		'''
		# A list of Decision Trees
		self.trees = []
		# A list of which features are use in each tree
		self.trees_features = []
		number_samples, number_features = X.shape[:2]
		for _ in range(self.nb_trees):
			# Random sampling with replacement for rows
			rows_index = self.rnd.choice(np.arange(0, number_samples), size = number_samples, replace = True)
			# Random sampling without replacement for cols (Choosing k features)
			cols_index = self.rnd.choice(np.arange(0, number_features), size = self.nb_features, replace = False)
            # Get subset
			subset_X, subset_y = (
				(X[rows_index.tolist(), :][:, cols_index.tolist()]).copy(), 
				y[rows_index.tolist()].copy()
			)
			tree = DecisionTreeClassifier(**kwargs).fit(subset_X, subset_y)
			self.trees.append(tree)
			self.trees_features.append(cols_index)
		return self
    
	def predict(self, X):
		'''
			Make predictions on dataset
			Paramters:
				X: numpy.ndarray of shape (n_samples, n_features)
					Testing set
			Returns: numpy.ndarray of shape (n_samples,)
				Prediction from testing set
        '''
		assert self.trees is not None, 'Random Forest object must fit to the training set first before making prediction'

		voteSamples = []
		for i, tree in enumerate(self.trees):
			features = self.trees_features[i]
			vote = tree.predict(X[:, features])
			voteSamples.append(vote.reshape(-1, 1))
		# Each column of voteSamples is a prediction from each tree
		# Each row of voteSamples is different "votes" of each tree for corresponding sample
		voteSamples = np.hstack(voteSamples)
		y_predict = []

		for votes in voteSamples:
			predict_result = self.forestVoting_(votes)
			y_predict.append(predict_result)
		return np.array(y_predict)
