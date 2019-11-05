import os, sys

import numpy as np

from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean, cosine
from numpy.linalg import norm

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

def dist_euclid(A, B):
    return norm(A-B)

def dist_sqr_euclid(A, B):
    return norm(A-B)**2

def dist_minkowski(A, B, k):
    return np.power(np.sum(np.subtract(A,B)**k), 1.0/k)

def correlation(A, B):
    return (pearson(A, B)+cos(A, B))*(np.sum((A-B)**2))/(len(A)**2)