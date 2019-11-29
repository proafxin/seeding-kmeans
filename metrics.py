import os, sys

import numpy as np

from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
from numpy.linalg import norm
from numpy import dot, array

def pearson(A, B):
    a = norm(A)
    b = norm(B)
    
    if a == 0 or b == 0.0:
        raise ValueError('Zero vectors in input')
    corr = pearsonr(A, B)
    corr_scaled = (corr[0]+1)/2.0
    return corr_scaled**2

def cos(A, B):
    a = norm(A)
    b = norm(B)
    if a == 0.0 or b == 0.0:
        raise ValueError('Zero vector in input')
    corr = cosine(A, B)
    corr_scaled = (corr+1.0)/2.0
    return corr_scaled**2

def distance_euclid(A, B):
    return norm(A-B)

def distance_squared(A, B):
    return dot(A-B, A-B)

def dist_minkowski(A, B, k):
    return np.power(np.sum(np.subtract(A,B)**k), 1.0/k)

def correlation(A, B):
    return (pearson(A, B)+cos(A, B))*(np.sum((A-B)**2))/(len(A)**2)

def process(X):
    def get_total_distance(x):
        tot = 0
        for y in X:
            tot += distance_euclid(x, y)
        return tot
    X = sorted(X, key=lambda x: get_total_distance(x))
    X = array(X)
    return X