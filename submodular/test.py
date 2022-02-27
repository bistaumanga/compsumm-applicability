from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np 
from submodular.functions import *
from submodular.feature_based_functions import *
from submodular.comparative_functions import *
from submodular.maximize import greedy

def test_functions():
    X = np.random.normal(size = (50, 2))
    K = pairwise_kernels(X, metric = "rbf", gamma = 1.0)
    print(X.shape, K.shape)
    fl = FacilityLocation(K)
    cut = Cut(K, lambdaa = 2.0)
    mmd = MMD(K)
    mmdw = MMD_w(K, np.random.random(len(X)))

    Sfl = greedy(fl)
    Scut = greedy(cut)
    Smmd = greedy(mmd)
    print(X[Sfl[0]], X[Scut[0]], X[Smmd[0]])
    Smmdw = greedy(mmdw)
    print(X[Smmdw[0]])

def test_features_based_functions():
    X = np.random.normal(size = (50, 2))
    K = pairwise_kernels(X, metric = "rbf", gamma = 1.0)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters = 8)
    km.fit(X)
    p = km.labels_
    print(X.shape, K.shape)
    lb = LinBilmes(K, p)
    Slb = greedy(lb)
    print(X[Slb[0]])

def test_comp():
    A = np.random.normal(size = (50, 2)) + np.array([-1, 1])
    B = np.random.normal(size = (60, 2)) + np.array([1, 1])
    X = np.vstack((A, B))
    K = pairwise_kernels(X, metric = "rbf", gamma = 1.0)
    y = np.array([-1] * 50 + [1] * 60)
    
    print(X.shape, K.shape)
    mmdc1 = Mmd_Comp(K, y, -1, 0.2)
    mmdc2 = Mmd_Comp(K, y, 1, 0.2)
    Slb1 = greedy(mmdc1, candidates = np.array([1] * 50 + [0] * 60))
    Slb2 = greedy(mmdc2, candidates = np.array([0] * 50 + [1] * 60))
    print(X[Slb1[0]], X[Slb2[0]])
    KA, KB = K[:50, :][:, :50], K[50:, :][:, 50:]
    KAB = K[:50, :][:, 50:]
    print(KA.shape, KB.shape, KAB.shape)
    mmdcw1 = MMD_Comp_w(KA, KB, KAB.T, np.ones(50), np.ones(60))
    mmdcw2 = MMD_Comp_w(KB, KA, KAB, np.ones(60), np.ones(50))
    Slb1 = greedy(mmdcw1)
    Slb2 = greedy(mmdcw2)
    print(X[Slb1[0]], X[Slb2[0]])

if __name__ == "__main__":
    test_comp()