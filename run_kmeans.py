from kmeans_clustering import KMeansClustering
from preprocess import process

import pandas as pd
import numpy as np
import timeit

from os.path import join, exists

class RunKmeans:
    file = None
    num_exp = None
    do_pca = False
    initlizations_ = [
        'random',
        'k-means++',
        'orss',
        'coc',
    ]
    n_clusters = 5

    def __init__(self, file, num_exp=20, n_clusters=5, do_pca=False):
        if type(file) != type('a'):
            raise ValueError('File path must be valid')
        elif exists(file):
            self.file = file
        else:
            raise FileNotFoundError('Check file path')
        self.num_exp = num_exp
        self.n_clusters = n_clusters
        self.do_pca = do_pca

    def __get_data(self):
        if self.file == None:
            raise ValueError('Input a valid file')
        df = pd.read_csv(self.file)
        return process(df, do_pca=self.do_pca)

    def run_kmeans(self):
        X = self.__get_data()
        inertias = {}
        times = {}
        iters = {}
        for init in self.initlizations_:
            inertias[init] = np.zeros(shape=(self.num_exp))
            times[init] = np.zeros(shape=(self.num_exp))
            iters[init] = np.zeros(shape=(self.num_exp))
            for i in range(self.num_exp):
                kmeans = KMeansClustering(
                    init=init,
                    n_init=10,
                    n_clusters=self.n_clusters,
                )
                start = timeit.default_timer()
                kmeans.fit(X)
                kmeans.predict(X)
                end = timeit.default_timer()
                # print(kmeans.cluster_centers_)
                inertias[init][i] = kmeans.inertia_
                times[init][i] = (end-start)
                iters[init][i] = kmeans.n_iter_
        return inertias, times, iters