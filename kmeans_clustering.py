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
    n_clusters = 5
    verbose = False
    cluster_centers_ = array([])
    labels_ = array([])
    random_state = None
    sse_ = 0.0
    wcss_ = 0.0
    iter_convergence_ = 10000

    def __init__(self, init, max_iter, n_clusters, verbose, random_state=None):
        if init is not None:
            self.init = init    
        if max_iter is not None:
            self.max_iter = math
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if verbose is not None:
            self.verbose = verbose
        if random_state is not None:
            self.random_state = random_state

    def __get_kmeans_centers(self, X, k):
        randoms = choice(
            range(X.shape[0]),
            size=k,
        )
        centers = []
        for r in randoms:
            centers.append(X[r].tolist())
        centers = array(centers)
        return centers
    
    def __get_kmeans_plus_plus_centers(self, X, k):
        r = randint(0, X.shape[0]-1)
        centers = []
        centers.append(X[r].tolist())
        for i in range(1, k):
            D = []
            for x in X:
                D.append(np.min(sum((centers-x)**2, axis=1)))
            D = array(D)
            probs = D/sum(D)
            probs_cuml = cumsum(probs)
            r = random()
            for (j, p) in enumerate(probs_cuml):
                if X[j].tolist() in centers:
                    continue
                if p > r:
                    centers.append(X[j].tolist())
                    break
        centers = array(centers)
        return centers

    def __get_ostrovsky_centers(self, X, k):
        D = []
        init_centers = []
        for i in range(X.shape[0]):
            r = randint(0, X.shape[0]-1)
            s = randint(0, X.shape[0]-1)
            if r == s:
                continue
            D.append(dot(X[r]-X[s], X[r]-X[s]))
            init_centers.append({'x':r, 'y':s})
        D = array(D)
        probs = D/sum(D)
        probs_cuml = cumsum(probs)
        r = random()
        centers = []
        for (i, p) in enumerate(probs_cuml):
            if p > r:
                x = init_centers[i]['x']
                y = init_centers[i]['y']
                centers.append(X[x].tolist())
                centers.append(X[y].tolist())
                break
        for i in range(1, k-1):
            D = []
            for x in X:
                # print(centers, x)
                D.append(np.min(np.sum((centers-x)**2, axis=1)))
            D = array(D)
            probs = D/sum(D)
            probs_cuml = cumsum(probs)
            r = random()
            for (j, p) in enumerate(probs_cuml):
                if X[j].tolist() in centers:
                    continue
                if p > r:
                    centers.append(X[j].tolist())
                    break
        centers = array(centers)
        return centers

    def __get_variance_based_centers(self, X, k):
        D = []
        init_centers = []
        for i in range(X.shape[0]):
            r = randint(0, X.shape[0]-1)
            s = randint(0, X.shape[0]-1)
            if r == s:
                continue
            D.append(dot(X[r]-X[s], X[r]-X[s]))
            init_centers.append({'x':r, 'y':s})
        D = array(D)
        probs = D/sum(D)
        probs = 1.0-probs
        probs_cuml = cumsum(probs)
        r = random()
        centers = []
        for (i, p) in enumerate(probs_cuml):
            if p > r:
                x = init_centers[i]['x']
                y = init_centers[i]['y']
                centers.append(X[x].tolist())
                centers.append(X[y].tolist())
                break
        # print(probs_cuml, centers)
        for i in range(1, k-1):
            V = []
            for x in X:
                D = []
                for c in centers:
                    D.append(dot(x-c, x-c))
                D = array(D)
                V.append(np.var(D))
            V = array(V)
            probs = V/sum(V)
            probs_cuml = cumsum(probs)
            r = random()
            for (j, p) in enumerate(probs_cuml):
                # print(X[j].tolist())
                if X[j].tolist() in centers:
                    continue
                if p > r:
                    centers.append(X[j].tolist())
                    break
        centers = array(centers)
        return centers 
        
    def fit(self, X):
        k = self.n_clusters
        if len(X) < 1:
            raise ValueError('Dataset must have size greater than 0')
        if self.random_state is not None:
            X = shuffle(X)
        if self.init == 'auto':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_centers(X, k)
        elif self.init == 'kmeans++':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_centers(X, k)
        elif self.init == 'kmeans':
            self.cluster_centers_ = self.__get_kmeans_centers(X, k)
        elif self.init == 'ostrovsky':
            self.cluster_centers_ = self.__get_ostrovsky_centers(X, k)
        elif self.init == 'variance':
            self.cluster_centers_ = self.__get_variance_based_centers(X, k)
        if self.verbose is not None:
            # print(X[:5])
            print('Initializtion', self.init)
            print(self.cluster_centers_)
        for c in self.cluster_centers_:
            for a in c:
                if type(a) != type(np.float64(1.0)):
                    raise ValueError('Centers not initialized properly')
        return self
    
    def predict(self, X):
        pass

