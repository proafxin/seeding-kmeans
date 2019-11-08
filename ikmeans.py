import os, random, math, cv2

import numpy as np

from random import sample, randint, random
from os.path import join
from math import sqrt
from numpy import append, array, subtract, sum, cumsum, mean
from numpy import dot, subtract, sqrt, average
from numpy.random import choice
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
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
    init = 'auto'

    def __init__(self, n_clusters=5, max_iter=300, verbose=0, n_jobs=0, init='auto'):
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if max_iter is not None:
            self.max_iter = max_iter
        if verbose is not None:
            self.verbose = verbose
            pass
        if init is not None:
            self.init = init
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

    def get_centers_ikmeans(self, X):
        n = X.shape[0]
        k = self.n_clusters
        r = randint(0, n)
        centers = array([X[r]])
        # s = randint(0, n)
        # centers = append(centers, [X[s]], axis=0)
        D = sum((X-centers[0])**2, axis=1)
        D = [sqrt(d) for d in D]
        D = array(D)
        median_dist = np.median(D)
        second_centers = []
        dists = cdist(X, centers, 'euclidean')
        for (x, d) in zip(X, dists):
            if d > median_dist:
                second_centers.append(x)
        second_centers = array(second_centers)
        s = choice(range(len(second_centers)))
        # print(s, second_centers[s])
        centers = append(
            centers,
            [second_centers[s]],
            axis=0,
        )
        for i in range(2, k):
            dists = cdist(X, centers, 'euclidean')
            # print(dists)
            D = array([])
            for x in X:
                D = append(
                    D,
                    np.min(sum((centers-x)**2)),
                )
            D /= sum(D)
            variances = np.var(dists, axis=1)
            tot_variance = sum(variances)
            # print(tot_variance, variances)
            probs = variances/tot_variance
            probs_cumulative = cumsum(probs)
            r = random()
            for (x, p) in zip(X, probs_cumulative):
                if p > r:
                    centers = append(
                        centers,
                        [x],
                        axis=0,
                    )
                    break
        return centers
    
    def get_centers_kmeans(self, X):
        rs = choice(
            range(X.shape[0]),
            size=self.n_clusters,
        )
        centers = array([X[rs[0]]])
        for i in range(1, self.n_clusters):
            # print(X[rs[i]], centers)
            centers = append(
                centers,
                [X[rs[i]]],
                axis=0,
            )
        return centers

    def fit(self, X):
        centers = array([])
        if self.__check_validity(X) is not True:
            raise ValueError('Clean up dataset')
        if self.init == 'auto' or self.init == 'k-means++':
            centers = self.get_centers_kmeans_plus(X)
        elif self.init == 'ikmeans':
            centers = self.get_centers_ikmeans(X)
        elif self.init == 'kmeans':
            centers = self.get_centers_kmeans(X)
        else:
            raise ValueError('init value not recognized')
        k = self.n_clusters
        if self.verbose == 1:
            print('Iterations:', self.max_iter)
            print('Number of clusters:', self.n_clusters)
            print('Centers:')
            print(centers)
        inertia = 0
        for iter in range(self.max_iter):
            clusters = {}
            for i in range(k):
                clusters[i] = []
            cur_inertia = 0
            for x in X:
                dists = sum((centers-x)**2, axis=1)
                min_dist = np.min(dists)
                cur_inertia += min_dist
                for (i, dist) in enumerate(dists):
                    if dist == min_dist:
                        clusters[i].append(x)
                        break
            center_dist = 0
            for i in range(k):
                cur_center = centers[i]
                centers[i] = mean(clusters[i], axis=0)
                center_dist += norm(cur_center-centers[i])
                # print(cur_center, centers[i], center_dist)
            if iter > 1 and center_dist == 0.0:
                print('Iterations for convergence:', iter)
                break
            inertia = cur_inertia
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
                    