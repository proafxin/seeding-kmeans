import os, random, math, cv2

import numpy as np

from random import sample, randint, random
from os.path import join
from math import sqrt
from numpy import append, array, subtract, sum, cumsum, mean
from numpy import dot, subtract, sqrt, average
from numpy.random import choice, shuffle
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from metrics import cos, pearson, correlation

class KMeansClustering():
    init = 'auto'
    max_iter = 100
    n_cluster = 5
    verbose = False
    cluster_centers_ = array([])
    labels_ = array([])
    random_state = None
    sse = 0.0
    wcss_ = 0.0

    def __init__(self, init, max_iter, n_cluster, verbose, random_state):
        if init is not None:
            self.init = init    
        if max_iter is not None:
            self.max_iter = math
        if n_cluster is not None:
            self.n_cluster = n_cluster
        if verbose is not None:
            self.verbose = verbose
        if random_state is not None:
            self.random_state = random_state

    def __get_kmeans_centers(self, X, k):
        randoms = choice(
            range(X.shape[0]),
            size=k,
        )
        centers = array([])
        for r in randoms:
            centers = append(
                centers,
                [X[r]],
                axis=0,
            )
        return centers
    
    def __get_kmeans_plus_centers(self, X, k):
        r = randint(0, X.shape[0])
        centers = array([X[r]])
        for i in range(1, k):
            for x in X:
                D = np.min(sum(centers-x)**2, axis=1)
            probs = D/D.sum()
            probs_cuml = cumsum(probs)
            r = random()
            for (i, p) in enumerate(probs_cuml):
                if p > r:
                    centers = append(
                        centers,
                        [X[i]],
                        axis=0,
                    )
        return centers

    def fit(self, X):
        k = self.n_cluster
        if len(X) < 1:
            raise ValueError('Dataset must have size greater than 0')
        if self.random_state is not None:
            X = shuffle(X)
        if self.init == 'auto':
            self.cluster_centers_ = self.__get_kmeans_plus_centers(X, k)
        elif self.init == 'kmeans++':
            self.cluster_centers_ = self.__get_kmeans_plus_centers(X, k)
        elif self.init == 'kmeans':
            self.cluster_centers_ = self.__get_kmeans_centers(X, k)
        
        return self

