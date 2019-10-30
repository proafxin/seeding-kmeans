import numpy as np

from os.path import join
from scipy.stats import pearsonr, cosine
from math import sqrt

def distance(A, B):
    total = 0
    for (a, b) in zip(A, B):
        diff = (a-b)
        total += diff*diff
    return sqrt(total)

def pearson(A, B):
    coeff = pearsonr(A, B)
    return (coeff+1.0)/2.0

def similarity(A, B):
    p = pearson(A, B)
    c = cosine(A, B)
    return (p+c)/2.0

class KMeans:
    n_clusters = 3
    distance = 'auto'
    labels = []
    
    def __init__(self, n_clusters, distance):
        self.n_clusters = n_clusters
        self.distance = distance
        self.labels = []
    
    def clusterize(self, X, k):
        n = len(X)
        labels = []
        if (n < 3):
            labels.append(0)
            labels.append(2)
            self.labels = labels
        else:
            centers = []
            while (len(centers) < k):
                centers.append(X[0])
