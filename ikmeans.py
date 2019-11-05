import os, random, math, cv2

import numpy as np

from random import sample, randint, random
from os.path import join
from math import sqrt
from numpy import append, array, subtract, sum, cumsum
from numpy import dot, subtract, sqrt, average
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from cv2 import convexHull
from scipy.spatial import ConvexHull
from metrics import cos, pearson, correlation

kmeans = KMeans()

class IKMeans:
    cluster_centers_ = array([])
    labels_ = array([])
    inertia_ = 0.0
    n_clusters = 5
    max_iter = 300
    verbose = 0
    n_jobs = 0

    def __init__(self, n_clusters=5, max_iter=300, verbose=0, n_jobs=0):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if max_iter is not None:
            self.max_iter = max_iter
        if verbose is not None:
            self.verbose = verbose
            pass
        pass
    
    def __check_validity(self, X):
        for a in X:
            for x in a:
                if x is None or np.isnan(x) is True:
                    raise ValueError('Nan or none value in dataset')
                if np.isinf(x) is True or np.isneginf(x) is True:
                    raise ValueError('Infinite value in dataset')
                if type(x) != type(np.float64(1.0)) and type(x) != type(np.int64(1)):
                    print(x, type(x))
                    raise TypeError('Values can be either int or float')
                if len(X) < 1:
                    raise ValueError('Dataset must have size > 0')
                pass
            pass
        return True
    
    def get_centers_kmeans_plus(self, X):
        n = X.shape[0]
        k = self.n_clusters
        r = randint(0, n)
        centers = array([X[r]])
        for i in range(1, k):
            D = array([])
            for x in X:
                d = []
                for c in centers:
                    d.append(np.sum((c-x)**2))
                    # d.append(correlation(c, x))
                D = append(
                    D,
                    np.min(d),
                )
            probs = D/sum(D)
            probs_cumulative = cumsum(probs)
            r = random()
            for (x, p) in zip(X, probs_cumulative):
                if p > r:
                    # print(x)
                    centers = append(
                        centers,
                        [x],
                        axis=0,
                    )
                    break
        return centers

    def fit(self, X):
        if self.__check_validity(X) is not True:
            raise ValueError('Clean up dataset')
        centers = self.get_centers_kmeans_plus(X)
        k = self.n_clusters
        if self.verbose == 1:
            print('Iterations:', self.max_iter)
            print('Number of clusters:', self.n_clusters)
            print('Centers:')
            print(centers)
        for iter in range(self.max_iter):
            iter += 0
            clusters = {}
            for i in range(k):
                clusters[i] = []
            for x in X:
                min_dist = 1.0e50
                cur = 0
                for (j, c) in enumerate(centers):
                    cur_dist = np.sum((c-x)**2)
                    # cur_dist = correlation(c, x)
                    if min_dist > cur_dist:
                        cur = j
                        min_dist = cur_dist
                clusters[cur].append(x)
            for i in range(k):
                # centers[i] = np.mean(clusters[i], axis=0)
                hull = ConvexHull(clusters[i])
                vertices = [clusters[i][j] for j in hull.vertices]
                center_kmeans = np.mean(clusters[i], axis=0)
                weights = array([])
                for vertex in vertices:
                    weights = append(
                        weights,
                        sum((center_kmeans-vertex)**2),
                    )
                    weights /= sum(weights)
                    weights = 1.0-weights
                # centers[i] = np.average(vertices, axis=0, weights=weights)
                centers[i] = np.mean(vertices, axis=0)
                # centers[i] = np.mean(clusters[i], axis=0)
        self.cluster_centers_ = centers
        return self
    
    def predict(self, X):
        k = self.n_clusters
        clusters = {}
        for i in range(k):
            clusters[i] = array([])
        labels = array([])
        inertia = 0.0
        for (i, x) in enumerate(X):
            dists = sum((self.cluster_centers_-x)**2, axis=1)
            min_dist = np.min(dists)
            inertia += min_dist
            for (i, dist) in enumerate(dists):
                if dist == min_dist:
                    labels = append(labels, i)
                    break

        self.inertia_ = inertia
        self.labels_ = labels
        if self.verbose == 1:
            print('Inertia:', self.inertia_)
        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
                    