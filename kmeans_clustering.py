import os, random, math, cv2

import numpy as np

from random import sample, randint, random
from os.path import join
from math import sqrt
from numpy import append, array, subtract, sum, cumsum, mean
from numpy import dot, subtract, sqrt, average, all
from numpy.random import choice, shuffle
from numpy.linalg import norm
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from metrics import cos, pearson, correlation, distance_squared

class KMeansClustering():
    init = 'auto'
    max_iter = 100
    n_clusters = 5
    verbose = False
    cluster_centers_ = array([])
    labels_ = array([])
    random_state = None
    sse_ = 0.0
    iter_convergence_ = 10000
    time = 0
    n_iters_ = 0
    best_inertia_ = 0
    centroid_ = None

    def __init__(self, init, max_iter, n_clusters, verbose, random_state=None):
        if init is not None:
            self.init = init    
        if max_iter is not None:
            self.max_iter = max_iter
        if n_clusters is not None:
            self.n_clusters = n_clusters
        if verbose is not None:
            self.verbose = verbose
        if random_state is not None:
            self.random_state = random_state
        self.n_iters_ = max_iter

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
    
    def __get_kmeans_pluss_centers_step2(self, X, k, centers):
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


    def __get_kmeans_plus_plus_centers(self, X, k):
        centers = []
        r = randint(0, X.shape[0]-1)
        centers.append(X[r].tolist())
        return self.__get_kmeans_pluss_centers_step2(X, k, centers)

    def __get_kmeans_plus_plus_initial_center(self, X):
        centers = []
        probs = []
        for x in X:
            d = x-self.centroid_
            probs.append(dot(d, d))
        probs /= sum(probs)
        probs_cuml = cumsum(probs)
        r = random()
        for (i, p) in enumerate(probs_cuml):
            if p > r:
                centers.append(X[i].tolist())
                break
        return centers

    def __get_kmeans_plus_plus_improved_centers(self, X, k):
        centers = self.__get_kmeans_plus_plus_initial_center(X)
        centers = self.__get_kmeans_pluss_centers_step2(X, k, centers)
        return centers
        
    def __get_ostrovsky_init_centers(self, X):
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
        return centers

    def __get_ostrovsky_centers(self, X, k):
        centers = self.__get_ostrovsky_init_centers(X)
        for i in range(1, k-1):
            D = []
            for x in X:
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

    def __get_kmeans_plus_plus_corrected_centers(self, X, k):
        r = randint(0, X.shape[0]-1)
        centers = array([X[r]])
        for i in range(1, k):
            # print(i, centers)
            mu_c = mean(centers, axis=0)
            D = []
            for x in X:
                D.append(dot(x-mu_c, x-mu_c))
            probs = D/sum(D)
            probs_cuml = cumsum(probs)
            r = random()
            for (j, p) in enumerate(probs_cuml):
                if p > r:
                    centers = append(
                        centers,
                        [X[j]],
                        axis=0,
                    )
                    break
        # print(centers, len(centers))
        return centers
    
    def __get_centroid_of_centers_based_centers(self, X, k):
        centers = self.__get_kmeans_plus_plus_initial_center(X)
        centers = array(centers)
        for i in range(1, k):
            mu_c = mean(centers, axis=0)
            probs = []
            for x in X:
                diff = x-mu_c
                probs.append(dot(diff, diff))
            probs = probs/sum(probs)
            probs_cuml = cumsum(probs)
            r = random()
            for (j, p) in enumerate(probs_cuml):
                if p > r:
                    centers = append(
                        centers,
                        [X[j]],
                        axis=0,
                    )
                    break
        return centers
        
    def __get_variance_based_centers(self, X, k):
        centers = self.__get_ostrovsky_init_centers(X)
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
            probs = probs
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

    def __get_tolerance(self, X, tol):
        variances = np.var(X, axis=0)
        return mean(variances)*tol

    def __converge_centers(self, X, centers):
        for i in range(self.max_iter):
            C = {}
            for i in range(self.n_clusters):
                C[i] = []
            inertia = 0
            for x in X:
                D = []
                for c in centers:
                    D.append(dot(c-x, c-x))
                dist_min = min(D)
                inertia += dist_min
                for (i, d) in enumerate(D):
                    if d == dist_min:
                        C[i].append(x)
                        break
            for i in range(self.n_clusters):
                if len(C[i]) > 0:
                    centers[i] = mean(C[i], axis=0)
        return centers
        
    def fit(self, X):
        k = self.n_clusters
        self.centroid_ = mean(X, axis=0)
        if len(X) < 1:
            raise ValueError('Dataset must have size greater than 0')
        if self.random_state is not None:
            X = shuffle(X)
        if self.init == 'auto':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_centers(X, k)
        elif self.init == 'kmeans++_corrected':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_corrected_centers(X, k)
        elif self.init == 'kmeans++':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_centers(X, k)
        elif self.init == 'kmeans':
            self.cluster_centers_ = self.__get_kmeans_centers(X, k)
        elif self.init == 'ostrovsky':
            self.cluster_centers_ = self.__get_ostrovsky_centers(X, k)
        elif self.init == 'variance':
            self.cluster_centers_ = self.__get_variance_based_centers(X, k)
        elif self.init == 'kmeans++_improved':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_improved_centers(X, k)
        elif self.init == 'kmeans++_corrected':
            self.cluster_centers_ = self.__get_kmeans_plus_plus_corrected_centers(X, k)
        elif self.init == 'coc':
            self.cluster_centers_ = self.__get_centroid_of_centers_based_centers(X, k)
        else:
            raise ValueError('init not defined')
        for c in self.cluster_centers_:
            for a in c:
                if type(a) != type(np.float64(1.0)):
                    raise ValueError('Centers not initialized properly')
        if self.verbose is True:
            print('Initializtion', self.init)
            print(self.cluster_centers_)
        self.cluster_centers_ = self.__converge_centers(X, self.cluster_centers_)
        if self.verbose is True:
            print(self.init, 'centers after fitting')
            print(self.cluster_centers_)
        return self

    def predict(self, X):
        labels = []
        self.sse_ = 0
        for x in X:
            D = []
            for c in self.cluster_centers_:
                D.append(dot(c-x, c-x))
            dist_min = min(D)
            self.sse_ += dist_min
            for (i, d) in enumerate(D):
                if d == dist_min:
                    labels.append(i)
                    break
        self.labels_ = labels
        return labels