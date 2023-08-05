import random

import numpy as np

from random import randint
from numpy import sum, cumsum, mean
from numpy import dot, all, float64, zeros
from numpy.random import choice

# from sklearn.cluster import KMeans
# from metrics import cos, pearson, correlation, distance_squared

from utils import threadpool_limits


class KMeansClustering:
    init = None
    n_init = None
    max_iter = None
    n_clusters = None
    verbose = None
    labels_ = None
    cluster_centers_ = None
    inertia_ = None
    n_iter_ = None

    def __init__(
        self,
        init="random",
        n_init=10,
        n_clusters=5,
        verbose=False,
        max_iter=100,
    ):
        self.verbose = verbose
        self.init = init
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

    def __get_kmeans(self, X):
        ar = [i for i in range(X.shape[0] - 1)]
        R = choice(ar, size=self.n_clusters)
        for i, r in enumerate(R):
            self.cluster_centers_[i] = X[r]

    def __get_most_probable(self, X, D):
        if abs(sum(D)) < 1e-15:
            print("Invalid distances")
            return None
        probs = D / sum(D)
        probs_cuml = cumsum(probs)
        r = random()
        for i, p in enumerate(probs_cuml):
            if p > r:
                return X[i]

    def __get_kmeans_plus_plus_initial(self, X):
        mu = mean(X, axis=0)
        D = []
        for x in X:
            y = x - mu
            D.append(dot(y, y))
        self.cluster_centers_[0] = self.__get_most_probable(X, D)

    def __get_kmeans_plus_plus_rest(self, X):
        for i in range(1, self.n_clusters):
            D = []
            for x in X:
                dists = []
                for c in self.cluster_centers_[:i]:
                    y = c - x
                    dists.append(dot(y, y))
                D.append(min(dists))
            self.cluster_centers_[i] = self.__get_most_probable(X, D)

    def __get_kmeans_plus_plus(self, X):
        r = randint(0, X.shape[0] - 1)
        self.cluster_centers_[0] = X[r]
        self.__get_kmeans_plus_plus_rest(X)

    def __get_orss(self, X):
        self.__get_kmeans_plus_plus_initial(X)
        self.__get_kmeans_plus_plus_rest(X)

    def __get_coc(self, X):
        r = randint(0, X.shape[0] - 1)
        self.cluster_centers_[0] = X[r]
        for i in range(1, self.n_clusters):
            mu = mean(self.cluster_centers_[:i], axis=0)
            D = []
            for x in X:
                y = x - mu
                D.append(dot(y, y))
            self.cluster_centers_[i] = self.__get_most_probable(X, D)

    def __initialize_centers(self, X):
        k = self.n_clusters
        self.cluster_centers_ = zeros(shape=(k, len(X[0])), dtype=float64)
        init = self.init
        if init == "random" or init == "auto":
            self.__get_kmeans(X)
        elif init == "k-means++":
            self.__get_kmeans_plus_plus(X)
        elif init == "coc":
            self.__get_coc(X)
        elif init == "orss":
            self.__get_orss(X)
        else:
            raise ValueError("Initialization not supported.")
        if self.verbose is True:
            print(self.cluster_centers_)

    def __converge(self, X):
        with threadpool_limits(limits=1, user_api="blas"):
            for i in range(self.max_iter):
                centers_old = self.cluster_centers_.copy()
                C = {}
                for j in range(self.n_clusters):
                    C[j] = []
                inertia = 0.0
                for x in X:
                    D = []
                    for c in self.cluster_centers_:
                        y = c - x
                        D.append(dot(y, y))
                    d_min = min(D)
                    inertia += d_min
                    for j, d in enumerate(D):
                        if d == d_min:
                            C[j].append(x.tolist())
                            break
                for j in range(self.n_clusters):
                    if len(C[j]) < 1:
                        continue
                    self.cluster_centers_[j] = mean(C[j], axis=0)
                if all(centers_old == self.cluster_centers_):
                    self.n_iter_ = i
                    self.inertia_ = inertia
                    break

    def fit(self, X):
        self.__initialize_centers(X)
        for c in self.cluster_centers_:
            for a in c:
                if isinstance(a, np.float64):
                    continue
                elif isinstance(a, np.int64):
                    continue
                else:
                    raise TypeError(a, " is neither int nor float")
        # print(self.cluster_centers_)
        self.__converge(X)
        return self

    def predict(self, X):
        self.labels_ = zeros(shape=(X.shape[0]))
        for i, x in enumerate(X):
            D = []
            for c in self.cluster_centers_:
                y = x - c
                D.append(dot(y, y))
            d_min = min(D)
            for j, d in enumerate(D):
                if d == d_min:
                    self.labels_[i] = j
                    break
        return self.labels_

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
