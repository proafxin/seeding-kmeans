from kmeans_clustering import KMeansClustering
from preprocess import process

import pandas as pd
import numpy as np

from os.path import join, exists

class RunKmeans:
    file = None
    num_exp = None
    initlizations_ = [
        'random',
        'k-means++',
        'orss',
        'coc',
        'var',
    ]
    n_clusters = 5

    def __init__(self, file, num_exp=20, n_clusters=5):
        if type(file) != type('a'):
            raise ValueError('File path must be valid')
        elif exists(file):
            self.file = file
        else:
            raise FileNotFoundError('Check file path')
        self.num_exp = num_exp
        self.n_clusters = n_clusters

    def __get_data(self):
        if self.file == None:
            raise ValueError('Input a valid file')
        df = pd.read_csv(self.file)
        return process(df)

    def run_kmeans(self):
        X = self.__get_data()
        inertias = {}
        times = {}
        iters = {}
        for init in self.initlizations_:
            print(init)
            inertias[init] = np.zeros(shape=(self.num_exp))
            times[init] = np.zeros(shape=(self.num_exp))
            iters[init] = np.zeros(shape=(self.num_exp))
            for i in range(self.num_exp):
                kmeans = KMeansClustering(
                    init=init,
                    n_init=10,
                    n_clusters=self.n_clusters,
                )
                kmeans.fit(X)
                inertias[init][i] = kmeans.inertia_
                times[init][i] = 0
                iters[init][i] = kmeans.n_iter_
                labels = kmeans.predict(X)
        return inertias, times, iters, labels