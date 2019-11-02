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

def minkowski(A, B, p):
    d = subtract(A, B)
    d = np.power(d, p)
    k = len(d)
    if k < 1:
        return 0
    d = np.sum(d)
    d /= k
    d = np.power(d, 1.0/k)
    return d

def distance_sqrd(A, B):
    d = subtract(A, B)
    return dot(d, d)

def pearson(A, B):
    # print(A, B)
    a = norm(A)
    b = norm(B)
    if a is 0.0 and b is 0.0:
        return 1.0
    elif a is 0.0 or b is 0.0:
        return 0.0
    else:
        corr = pearsonr(A, B)
        sim = (corr[0]+1.0)/2.0
        dis = 1.0-sim
        return dis

def cosine(A, B):
    a = norm(A)
    b = norm(B)
    ab = dot(A, B)
    if a is 0.0 and b is 0.0:
        return 0.0
    elif a is 0.0 or b is 0.0:
        return 1.0
    else:
        c = ab/(a*b)
        sim = (c+1.0)/2.0
        dis = 1.0-sim
        return dis
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
    labels_ = []
    n_iters = 500
    cluster_centers_ = []
    verbose = False

    def __init__(self, n_clusters, n_iters, verbose=False):
        if self.n_clusters > 1:
            self.n_clusters = n_clusters
        self.labels_ = []
        if n_iters > 0:
            self.n_iters = n_iters
        self.cluster_centers_ = []
        if verbose != None:
            self.verbose = verbose
        else:
            self.verbose = False

    def check_validity(self, X):
        for x in X:
            for a in x:
                if type(a) != type(np.float64(1.0)) and type(a) != type(np.int64(1)):
                    print(x, a, type(a))
                    raise(TypeError('Arrays should contain either integers or floats'))
                if np.isnan(a):
                    raise ValueError('Dataset can not have nan values')
                elif np.isinf(a):
                    raise ValueError('Dataset can not have infinite values')
                elif np.isneginf(a):
                    raise ValueError('Dataset can not have negative infinite values')
        return True

    def get_centers_d_sqrd(self, X, k):
        n = len(X)
        c0 = randint(0, n)
        cluster_centers_ = np.array([X[c0]])
        for i in range(1, k):
            D = np.array([])
            for x in X:
                d = np.array([])
                for c in cluster_centers_:
                    # print(c, x)
                    d = append(
                        d,
                        # distance_sqrd(c, x),
                        closeness(c, x),
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
                    cluster_centers_ = append(
                        cluster_centers_,
                        [X[a]],
                        axis=0,
                    )
                    break
                pass
            pass
        return cluster_centers_

    def fit(self, X):
        k = self.n_clusters
        n = len(X)
        if n <= k:
            raise ValueError('Number of data points should be higher than number of clusters')
        if self.check_validity(X) != True:
            raise ValueError('Clean your dataset')
        if self.verbose is True:
            print(X[:5])
            print('Number of data points:', len(X))
            print('number of clusters:', self.n_clusters, 'number of iterations:', self.n_iters)
        # print(c0, cluster_centers_)
        cluster_centers_ = self.get_centers_d_sqrd(X, k)
        self.cluster_centers_ = cluster_centers_
        if self.check_validity(self.cluster_centers_) != True:
            raise ValueError('cluster_centers_ are not valid')
        if self.verbose is True:
            print('Initial cluster_centers_:')
            print(cluster_centers_)
        for i in range(self.n_iters):
            C = {}
            for i in range(k):
                C[i] = []
            for x in X:
                min_dist = 1.0e50
                cluster = 0
                for i in range(k):
                    cur = closeness(x, cluster_centers_[i])
                    if cur < 0.0:
                        print(x, i, cur)
                        raise ValueError('Closeness found negative')
                    # cur = distance_sqrd(x, cluster_centers_[i])
                    if cur < min_dist:
                        min_dist = cur
                        cluster = i
                C[cluster].append(x)
            for i in range(k):
                cluster_centers_[i] = np.mean(C[i], axis=0)
        self.cluster_centers_ = cluster_centers_
        return self

    def predict(self, X):
        n = len(X)
        k = len(self.cluster_centers_)
        labels_ = []
        if (n < 3) or (k < 1):
            self.labels_ = [i for i in range(n)]
        for x in X:
            cluster = 0
            min_dist = 1e50
            for i in range(k):
                cur = closeness(x, self.cluster_centers_[i])
                # cur = distance_sqrd(x, self.cluster_centers_[i])
                if cur < min_dist:
                    min_dist = cur
                    cluster = i
            labels_.append(cluster)
        self.labels_ = labels_
        return labels_
    
    def fit_predict(self, X):
        self.cluster_centers_ = self.fit(X)
        self.labels_ = self.predict(X)
        return self.labels_

class ErrorChecker:
    dist_tot = 0
    cluster_centers_ = []
    X = []
    labels_ = []
    n_clusters = 0
    def __init__(self, X, cluster_centers_, labels_):
        self.cluster_centers_ = cluster_centers_
        self.X = X
        self.n_clusters = len(cluster_centers_)
        self.labels_ = labels_
    def potential_function(self, X=None, n_clusters=None):
        if X is not None:
            self.X = X
        if n_clusters is not None:
            assert(len(self.cluster_centers_) == n_clusters)
        n = len(self.X)
        assert(n > 0)
        assert(len(self.cluster_centers_) > 2)
        assert(len(self.labels_) == n)
        dist_tot = 0
        # print(self.X[:5])
        # print(self.cluster_centers_)
        for x in self.X:
            for c in self.cluster_centers_:
                # print(c, x)
                dist_tot += distance_sqrd(c, x)
                pass
            pass
        return dist_tot
            