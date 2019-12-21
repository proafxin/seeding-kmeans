import os, timeit

import pandas as pd

from kmeans_clustering import KMeansClustering
from preprocess import process

class KMeansSimulator():
    file = None
    __initializations = [
        'random', 
        'k-means++', 
        'coc',
        # 'variance',
        'orss',
    ]
    init = None
    num_exp = None
    inertia_ = None
    inertia_with_pca_ = None
    n_iter_with_pca_ = None
    n_iter_ = None
    n_cluster = None
    time_ = None
    time_with_pca_ = None
    
    def __init__(self, file, num_exp=20, n_cluster=[5],):
        if file is None or file == '':
            raise ValueError('File name can not be empty')
        elif os.path.exists(file):
            self.file = file
        else:
            raise FileNotFoundError('File '+file+' not found')
        self.num_exp = num_exp
        self.n_cluster = n_cluster
        self.inertia_ = {}
        self.inertia_with_pca_ = {}
        self.n_iter_ = {}
        self.n_iter_with_pca_ = {}
        self.time_ = {}
        self.time_with_pca_ = {}
    
    def __get_data(self):
        df = pd.read_csv(self.file)
        X = process(
            df=df,
        )
        X_pca = process(
            df=df,
            do_pca=True,
        )
        return (X, X_pca)

    def run_kmeans(self):
        X, X_pca = self.__get_data()
        for num in self.n_cluster:
            # print(num)
            self.inertia_[num] = {}
            self.inertia_with_pca_[num] = {}
            self.time_[num] = {}
            self.time_with_pca_[num] = {}
            self.n_iter_[num] = {}
            self.n_iter_with_pca_[num] = {}
            for init in self.__initializations:
                self.inertia_[num][init] = []
                self.inertia_with_pca_[num][init] = []
                self.time_with_pca_[num][init] = []
                self.time_[num][init] = []
                self.n_iter_[num][init] = []
                self.n_iter_with_pca_[num][init] = []
                for i in range(self.num_exp):
                    algo = KMeansClustering(
                        init=init,
                        n_clusters=num,
                        max_iter=200,
                        verbose=False,
                        n_init=10,
                    )
                    start = timeit.default_timer()
                    algo = algo.fit(X)
                    algo.predict(X)
                    end = timeit.default_timer()
                    self.inertia_[num][init].append(algo.inertia_)
                    self.n_iter_[num][init].append(algo.n_iter_)
                    self.time_[num][init].append(end-start)
                    start = timeit.default_timer()
                    algo.fit(X_pca)
                    algo.predict(X_pca)
                    end = timeit.default_timer()
                    self.inertia_with_pca_[num][init].append(algo.inertia_)
                    self.n_iter_with_pca_[num][init].append(algo.n_iter_)
                    self.time_with_pca_[num][init].append(end-start)