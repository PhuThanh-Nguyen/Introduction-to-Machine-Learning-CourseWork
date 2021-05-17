class KMeansCluster:
	def __init__(self, number_clusters = None):
		self.number_clusters = number_clusters 

	def __getDistanceMatrix__(self, X, clusters):
		'''
			Given K clusters (c_1, c_2,...,c_K) and n data points (x_1, x_2,...,x_n)
			where each data point is a row in X.
			Distance matrix (n, K) has the form:
			[
					[d(x_1, c_1), d(x_1, c_2), ..., d(x_1, c_K)],
					[d(x_2, c_1), d(x_2, c_2), ..., d(x_2, c_K)],
					...
					[d(x_n, c_1), d(x_n, c_2), ..., d(x_n, c_K)]
				]
		'''
		number_obsv, dimension = X.shape[:2]
		dist = [
			np.linalg.norm(X - cluster.reshape((1, dimension)), axis = 1)
			.reshape((number_obsv, 1)) for cluster in clusters
		]
		return np.hstack(dist)

	def __getIndexCluster__(self, distanceMatrix):
		return distanceMatrix.argmin(axis = 1)

	def __WCV__(self, index_cluster, X):
		wcv = 0
		for i in np.unique(index_cluster):
			points_in_cluster = X[index_cluster == i]
			wcv += pairwiseDist(points_in_cluster).sum() * 1/points_in_cluster.shape[0]
		return wcv

	def fit(self, X, init_clusters = None, maxiter = 100, random_state = 0):
		number_obsv, dimension = X.shape[:2]

		#Initialize clusters centroid
		if init_clusters is None:
		# If not pre-assign, randomly choose K (number clusters) data points in X
			rnd = np.random.RandomState(random_state)
			shuffle_index = rnd.choice(np.arange(number_obsv), size = self.number_clusters, replace = False)
			self.clusters = X[shuffle_index].astype(float)
		else:
			# init_clusters must have the form (K, p), each row is a centroid in p-dimension
			assert init_clusters.shape == (self.number_clusters, dimension)
			self.clusters = init_clusters.copy()

		# Initial distance of each point to each cluster
		distance_matrix = self.__getDistanceMatrix__(X, self.clusters)
		# Initial label of each point
		index_cluster = self.__getIndexCluster__(distance_matrix)

		for _ in range(maxiter):
			# Update each centroid
			for i, centroid in enumerate(self.clusters):
				new_centroid = X[index_cluster == i].mean(axis = 0)
				# If new centroid doesn't change much from current centroid then quit
				if np.allclose(new_centroid, centroid):
					return self.__WCV__(index_cluster, X)
				self.clusters[i, :] = new_centroid
			# Update distance of each point to each cluster and label of each point
			distance_matrix = self.__getDistanceMatrix__(X, self.clusters)
			index_cluster = self.__getIndexCluster__(distance_matrix)

		return self.__WCV__(index_cluster, X)

	def predict(self, X):
		# Predict label of one sample -> Reshape it into (1, p) matrix
		if len(X.shape) == 1:
			X = X.reshape((1, X.shape[0]))
		distance_matrix = self.__getDistanceMatrix__(X, self.clusters)
		index_cluster = self.__getIndexCluster__(distance_matrix)
		return index_cluster
