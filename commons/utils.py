import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.metrics.pairwise import pairwise_kernels
from functools import lru_cache
import scipy as sp
import logging
import os
from datetime import datetime, timedelta

DEFAULT_FORMATTER = logging.Formatter(
                '%(asctime)s %(levelname)s [%(name)s:%(lineno)d]=> %(message)s', "%b-%d %H:%M:%S")

def apply_file_handler(logger, path):
    logger.handlers = []
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(file_handler)
    return logger

def get_logger(name, level = None, handler = None):
    logger = logging.getLogger(name)
    logger.propagate = False
    if not logger.handlers:
        if handler is None:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(DEFAULT_FORMATTER)
            logger.addHandler(stream_handler)
        else:
            logger.addHandler(handler)
        logger.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    return logger

def split_dates(start, end, intv, fmt = "%Y-%m-%d"):
    start = datetime.strptime(start,fmt)
    end = datetime.strptime(end,fmt)
    step = timedelta(days=intv)
    curr = start
    while curr < end:
        yield(curr.strftime(fmt))
        curr += step
    yield(end.strftime(fmt))

def kmeanspp(X, n_clusters, dist = 'euclidean', seed = 0):
	c_idxs = [np.random.RandomState(seed).randint(X.shape[0])]
	for _ in range(1, n_clusters):
		D_x2 = cdist(X, X[c_idxs], metric = dist).min(axis = 1) ** 2
		p_x = D_x2 / np.sum(D_x2)
		c_idxs.append(p_x.argmax())
	return c_idxs

def summarize_kmeanspp(X, n_clusters, init_centroids = None):
    if not init_centroids:
    	init_centroids = kmeanspp(X, n_clusters = n_clusters)
    kmeans = KMeans(n_clusters = n_clusters, init = X[init_centroids])
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    nn = NN(n_neighbors = 1, algorithm = 'auto', metric = "euclidean")
    nn.fit(X)
    return nn.kneighbors(centroids)[1].squeeze().tolist()

class GramMatrix(object):
    """
        A kernel matrix that memorizes the mean of rows
    """
    def __init__(self, K):
        self.__data = K
        self.__sum = K.sum()
        # assert self.shape[0] == self.shape[1], self.shape

    def __repr__(self):
        return "[{:.4f}, {:.4f}, {:4f}]".format(self.__data.min(), np.median(self.__data), self.__data.max())
    
    def __getitem__(self, index):
        return GramMatrix(self.__data[index])

    def __call__(self, index = None):
        if index is None: 
            return self.__data
        else: 
            return self[index]

    def __len__(self):
        return len(self.__data)

    @property
    def shape(self):
        return self.__data.shape

    @property 
    def sum(self):
        return self.__sum

    @property
    def avg(self):
        return 1.0 * self.__sum /  ( len(self) * len(self) )

    def __del__(self):
        del self.__data

    def regularize(self, alpha):
        self.__data = self.__data + alpha * np.eye(self.__data.shape[0])

    @lru_cache(maxsize=10)
    def __compute__mean(self, rows = []):
        N = len(rows)
        if N == 0:
            return self.__data.mean(axis = 0)
        else:
            return self.__data[rows, :].mean(axis = 0)
    
    def mean(self, rows = []):
        return self.__compute__mean(tuple(rows))

    def diagonal(self):
        return self.__data.diagonal()

    @staticmethod
    def create(X, **kernelargs):
        return GramMatrix(pairwise_kernels(X, **kernelargs))
