import os, random, math

import numpy as np

from random import sample, randint
from os.path import join
from math import sqrt
from numpy import corrcoef, append
from numpy import dot, subtract, sqrt
from numpy.linalg import norm
from scipy.stats import pearsonr

def distance(A, B):
    d = subtract(A, B)
    d = dot(d, d)
    return sqrt(d)

def distance_sqrd(A, B):
    d = subtract(A, B)
    return dot(d, d)

def pearson(A, B):
    # print(A, B)
    a = norm(A)
    b = norm(B)
    if a*b == 0.0:
        if a+b == 0.0:
            return 1.0
        return 0.0
    corr = pearsonr(A, B)
    return (corr[0]+1.0)/2.0

def cosine(A, B):
    a = norm(A)
    b = norm(B)
    ab = dot(A, B)
    if a*b == 0.0:
        if a+b == 0.0:
            return 1.0
        return 0.0
    c = ab/(a*b)
    return (c+1.0)/2.0
def similarity(A, B):
    p = pearson(A, B)
    c = cosine(A, B)
    return (p+c)/2.0

def closeness(A, B):
    return similarity(A, B)*distance(A, B)

def get_randoms(n, k, weights):
    numbers = np.random.choice(
        [i for i in range(n)],
        size=k,
        p=weights,
    )
    return numbers
    
class SBKMeans:
    n_clusters = 3
    labels = []
    n_iters = 500
    centers = []

    def __init__(self, n_clusters, n_iters):
        if self.n_clusters > 1:
            self.n_clusters = n_clusters
        self.labels = []
        if n_iters > 0:
            self.n_iters = n_iters
        self.centers = []

    def check_validity(self, X):
        for x in X:
            for a in x:
                if np.isnan(a):
                    raise ValueError('Dataset can not have nan values')
                elif np.isinf(a):
                    raise ValueError('Dataset can not have infinite values')
                elif np.isneginf(a):
                    raise ValueError('Dataset can not have negative infinite values')
                elif norm(a) == 0.0:
                    raise ValueError('Datapoint can not have all 0. Remove such points')
        return True

    def fit(self, X):
        k = self.n_clusters
        n = len(X)
        if n <= k:
            raise ValueError('Number of data points should be higher than number of clusters')
        if self.check_validity(X) != True:
            raise ValueError('Clean your dataset')
        # print(X[:5])
        # print('Number of data points:', len(X))
        # print('number of clusters:', self.n_clusters, 'number of iterations:', self.n_iters)
        c0 = randint(0, n)
        centers = np.array([X[c0]])
        # print(c0, centers)
        for i in range(1, k):
            D = np.array([])
            for x in X:
                d = np.array([])
                for c in centers:
                    # print(c, x)
                    d = append(
                        d,
                        distance_sqrd(c, x),
                    )
                D = append(
                    D,
                    np.min(d),
                )
            for d in D:
                assert(d >= 0.0)
            if np.sum(D) > 0:
                probabilities = D/np.sum(D)
            else:
                raise ValueError('All distances are zero for ', i, 'th center')
            cdf = np.cumsum(probabilities)
            assert(len(cdf) == n)
            rand_float = random.random()
            # print('Probability array length:', len(probabilities), len(cdf))
            # print('Distance array length:', len(D))
            for (a, p) in enumerate(cdf):
                if p > rand_float and a < n:
                    # print(a, np.array(X[a]))
                    centers = append(
                        centers,
                        [X[a]],
                        axis=0,
                    )
                    break
                pass
            pass
        print('Initial centers:')
        print(centers)
        for i in range(self.n_iters):
            C = {}
            for i in range(k):
                C[i] = []
            for x in X:
                min_dist = 1.0e50
                cluster = 0
                for i in range(k):
                    cur = closeness(x, centers[i])
                    if cur < 0.0:
                        print(x, i, cur)
                        raise ValueError('Similarity negative')
                    # cur = distance_sqrd(x, centers[i])
                    if cur < min_dist:
                        min_dist = cur
                        cluster = i
                C[cluster].append(x)
            for i in range(k):
                centers[i] = np.mean(C[i], axis=0)
        self.centers = centers
        return centers
    def predict(self, X):
        n = len(X)
        k = len(self.centers)
        labels = []
        if (n < 3) or (k < 1):
            self.labels = [i for i in range(n)]
        for x in X:
            cluster = 0
            min_dist = 1e50
            for i in range(k):
                cur = closeness(x, self.centers[i])
                cur = distance_sqrd(x, self.centers[i])
                if cur < min_dist:
                    min_dist = cur
                    cluster = i
            labels.append(cluster)
        self.labels = labels
        return labels
    
    def fit_predict(self, X):
        self.centers = self.fit(X)
        self.labels = self.predict(X)
        return self.labels

class ErrorChecker:
    dist_tot = 0
    centers = []
    X = []
    labels = []
    n_clusters = 0
    def __init__(self, X, centers, labels):
        self.centers = centers
        self.X = X
        self.n_clusters = len(centers)
        self.labels = labels
    def potential_function(self, X=None, n_clusters=None):
        if X is not None:
            self.X = X
        if n_clusters is not None:
            assert(len(self.centers) == n_clusters)
        n = len(self.X)
        assert(n > 0)
        assert(len(self.centers) > 2)
        assert(len(self.labels) == n)
        dist_tot = 0
        # print(self.X[:5])
        # print(self.centers)
        for x in self.X:
            for c in self.centers:
                # print(c, x)
                dist_tot += distance_sqrd(c, x)
                pass
            pass
        return dist_tot
            