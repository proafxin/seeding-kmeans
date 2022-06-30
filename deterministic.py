"""Test deterministic version of KMeans"""

import numpy as np
import pandas as pd
from copy import deepcopy

from utils  import threadpool_limits

def dot(A, B):
    return np.einsum('ij,ij->i', A, B)

class DeterministicKMeans():
    def __init__(self, data, num_cluster) -> None:
        assert np.equal(num_cluster, int(num_cluster)) == True
        num_cluster = int(num_cluster)
        assert num_cluster > 1
        __first_center = None
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        assert isinstance(data, list) or isinstance(data, np.ndarray)
        for row in data:
            assert isinstance(row, list) or isinstance(data, np.ndarray)
        self.__data = data
        self.__num_clusters = num_cluster
        self.inertia_ =  None
        self.labels_ = None
        self.centers_ = None
        self.num_iters_ = None

    def _get_centroid(self, data):
        return np.mean(data, axis=0)

    def _get_new_centers(self, initial_centers, data):
        centroid = self._get_centroid(data=data)
        distances = np.subtract(self.__data, centroid)
        variances = dot(distances, distances)
        den = sum(variances)
        if np.equal(den, 0.0) == True:
            raise ValueError('All datapoints are equal')
        max_var = np.max(variances)
        for i, variance in enumerate(variances):
            if len(initial_centers) >= self.__num_clusters:
                break
            if np.equal(variance, max_var) == True:
                initial_centers.append(self.__data[i])
        return initial_centers

    def _get_initial_centers(self):
        with threadpool_limits(limits=1, user_api="blas"):
            initial_centers = self._get_new_centers([], self.__data)
            while len(initial_centers) < self.__num_clusters:
                # initial_centers = self._get_new_centers(initial_centers=initial_centers, data=initial_centers)
                probs = []
                for row in self.__data:
                    distances = np.subtract(initial_centers, row)
                    distances = dot(distances, distances)
                    dmin = np.min(distances)
                    probs.append(dmin)
                probmax = np.max(probs)
                for i, prob in enumerate(probs):
                    if np.equal(prob, probmax) == True:
                        initial_centers.append(self.__data[i])
                        break

        return initial_centers

    def fit(self):
        with threadpool_limits(limits=1, user_api="blas"):
            centers = self._get_initial_centers()
            num_iter = 0
            while True:
                centers_old = deepcopy(centers)
                C = {}
                for j in range(self.__num_clusters):
                    C[j] = []
                inertia = 0.0
                for row in self.__data:
                    distances = np.subtract(centers_old, row)
                    D = dot(distances, distances)
                    d_min = min(D)
                    inertia += d_min
                    for (j, distance) in enumerate(D):
                        if np.equal(distance, d_min) == True:
                            C[j].append(list(row))
                            break
                for j in range(self.__num_clusters):
                    if len(C[j]) < 1:
                        continue
                    centers[j] = np.mean(C[j], axis=0)
                num_iter += 1
                if np.array_equal(centers, centers_old):
                    self.inertia_ = inertia
                    self.centers_ = centers
                    break
            self.num_iters_ = num_iter
        return self

    def predict(self, points):
        if isinstance(points, pd.DataFrame):
            points = points.to_numpy()
        assert isinstance(points, np.ndarray) or isinstance(points, list)
        assert isinstance(points[0], list) or isinstance(points[0], np.ndarray)
        assert len(points[0]) == len(self.__data[0])
        labels = []
        for point in points:
            distances = self.centers_-point
            dmin = np.min(distances)
            for i, distance in enumerate(distances):
                if np.equal(distance, dmin) is True:
                    labels.append(i)
                    break
        self.labels_ = labels

        return self.labels_
