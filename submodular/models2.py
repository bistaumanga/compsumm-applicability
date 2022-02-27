from sklearn.metrics.pairwise import pairwise_kernels
from commons.utils import GramMatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import networkx as nx
from heapq import nlargest
from operator import itemgetter
from itertools import chain
import scipy as sp

import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.neighbors import NearestNeighbors as NN
from submodular.functions import *
from submodular.feature_based_functions import *
from submodular.comparative_functions import *
import submodular.llr as llr
from submodular.maximize import greedy as greedy_maximize
from collections import Counter
from commons.utils import get_logger
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.cluster import KMeans
from commons.utils import kmeanspp
from commons.functions import avg_f1 as scorer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# def _snap_vecs(X, y, vecs, mk):
#     clss = np.array(sorted(set(y)), dtype = np.int8)
#     idxs = []
#     for c, k in enumerate(clss):
#         ixs_c = np.where(y == k)[0]
#         nn = NN(n_neighbors = 1, algorithm = 'auto', metric = "euclidean")
#         nn.fit(X[ixs_c])
#         idxs.append(ixs_c[nn.kneighbors(vecs[c*mk: (c+1)*mk])[1].squeeze().tolist()])
#     return np.array(idxs).flatten().tolist()

class PandasSelector(TransformerMixin):
    """
    selects column for pandas data frames
    """
    def __init__(self, col_name, **kwargs):
        self.col_name = col_name
    
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.col_name)

    def fit(self, X, y = None, **kwargs):
        """
        dXf: pandas dataframe
        """
        return self
    
    def transform(self, X, y = None, **kwargs):
        """
        X: pandas dataframe
        """
        # print(X[self.col_name].values[:5], y)
        res = np.array(X[self.col_name].values.tolist()).squeeze()
        return res

class BoW(TransformerMixin, BaseEstimator):
    """
        Bag of Words or Bag of Entities with weightings of word/entities
        - `weights` can be `idf`: inverse doc frequency or `llr`: square-root LLR
        - `tf_binary`: True or False (use counts or binary rep for each doc/sentence)
        
        WARN: 
        - Do not use `weights` and `tf_binary` as hyperparams in gridsearch. To use, 
        implement `set_params` and override default params of TfIdfVectorizer. (see GraphCompKSVM)
    """

    def __init__(self, tf_binary = False, weights = None, eta = 2.58, **kwargs):
        self.tf_binary = tf_binary
        self.weights = weights
        self.eta = eta
        assert self.weights in {None, "idf", "llr", "tanh"}
        self._logger = get_logger(self.__class__.__name__)
        self.llr_weights = None
        self.bow_vectorizer = TfidfVectorizer(
            tokenizer = lambda x: x, ## we already have list of entities
            lowercase = False,
            binary = self.tf_binary,
            use_idf = self.weights == "idf",
            preprocessor = None,
            norm = None,
            min_df = 2,
            max_df = 0.2,
            max_features = 2000,
        )
    
    @property
    def vocab(self):
        return sorted(self.bow_vectorizer.vocabulary_, 
            key=self.bow_vectorizer.vocabulary_.get)
    
    @property
    def vocab_weights(self):
        if self.weights == "idf":
            return np.array([self.bow_vectorizer.idf_[ix] for ix, v \
                in enumerate(self.vocab)])
        elif self.weights in {"llr", "tanh"}:
            return self.llr_weights
        else:
            return np.array([1] * len(self.vocab))
    
    def __repr__(self):
        return "{}(tf={}, weights={})".format(
            self.__class__.__name__, 
            "binary" if self.tf_binary else "counts",
            self.weights
        )
    
    def fit(self, X, y = None, **kwargs):
        self._logger.debug("fit: data.shape {}".format(X.shape))
        self._logger.debug("fit: data[0] {}".format(X[0]))
        self.bow_vectorizer.fit(X)
        
        if self.weights in {"llr", "tanh"}:
            assert len(y) == X.shape[0], "must provide y"
            clss = sorted(set(y))
            assert len(clss) == 2, "only two class supported now"
            counters = []
            for c,k in enumerate(clss):
                idxs = np.where(y == k)[0]
                counters.append(Counter(chain(*X[idxs, ])))
            self._logger.debug(counters)
            srllr = llr.llr_compare(counters[0], counters[1])
            self.llr_weights = np.array([srllr[v] for v in self.vocab])

        return self
    
    def transform(self, X, y = None, **kwargs):
        self._logger.debug("transform: data.shape {}".format(X.shape))
        self._logger.debug("transform: data[0] {}".format(X[0]))
        bow = self.bow_vectorizer.transform(X)
        
        if self.weights == "llr":
            return bow * sp.sparse.diags(self.llr_weights, 0)
        elif self.weights == "tanh":
            return bow * sp.sparse.diags(np.tanh(self.llr_weights/self.eta) , 0)
        return bow
    
class Kernel(TransformerMixin, BaseEstimator):
    '''
    Stateful Kernel transformation for SKLearn pipelines
    useful for comparative summarisation pipelines
    '''
    def __init__(self, **kwargs):
        self._logger = get_logger(self.__class__.__name__)

    def fit(self, X, y = None, **kwargs):
        self._logger.debug("fit: data.shape {}".format(X.shape))
        self._logger.debug("fit: data[0] {}".format(X[0]))
        self.data = X
        return self
    
    def _compute_kernel(self, X):
        raise NotImplementedError("should implement transform")

    def transform(self, X, y = None, **kwargs):
        self._logger.debug("transform: data.shape {}".format(X.shape))
        self._logger.debug("transform: data[0] {}".format(X[0]))
        K = self._compute_kernel(X)
        degree = K.sum(axis=1)
        self._logger.debug("{} #nodes:{}, sparsity:{:.2f}, isolated:{:.2f}".format(
            self.__class__.__name__, K.shape[0], 1-(K > 0).sum()/(K.shape[0] * K.shape[1]), 
            1-( degree > 0).sum() / K.shape[0]
        ))
        return GramMatrix(K)

class RbfK(Kernel):
    def __init__(self, gamma = 0.11, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)

    def _compute_kernel(self, X):
        self._logger.debug("Computing RBF with gamma={}".format(self.gamma))
        return pairwise_kernels(X, self.data, metric = "rbf", gamma = self.gamma)

class ExpK(Kernel):
    def __init__(self, gamma = 0.11, **kwargs):
        self.gamma = gamma
        super().__init__(**kwargs)

    def _compute_kernel(self, X):
        self._logger.debug("Computing RBF with gamma={}".format(self.gamma))
        K = pairwise_kernels(X, self.data, metric = "cosine")
        K = np.exp(-self.gamma) * np.exp(K)
        return K

class LinearK(Kernel):
    def _compute_kernel(self, X):
        return pairwise_kernels(X, self.data, metric = "linear")

class CosineK(Kernel):
    def _compute_kernel(self, X, y = None):
        return pairwise_kernels(X, self.data, metric = "cosine")

class ThreshK(TransformerMixin):
    def __init__(self, upper = 1.0, lower = None):
        self.lower = lower
        self.upper = upper
        if self.lower is not None:
            assert self.upper > self.lower,  "invalid thresholds"
    
    def fit(self, X, y = None, **kwargs):
        self._logger.debug("fit: data.shape {}".format(X.shape))
        self._logger.debug("fit: data[0] {}".format(X[0]))
        return self
    
    def transform(self, X, y = None, **kwargs):
        self._logger.debug("transform: data.shape {}".format(X.shape))
        self._logger.debug("transform: data[0] {}".format(X[0]))
        X[X >= self.upper] = self.upper
        if self.lower is not None:
            X[X < self.lower] = self.lower
        return X

class UnGramMatrix(TransformerMixin):    
    def fit(self, X, y = None, **kwargs):
        return self
    
    def transform(self, X, y = None, **kwargs):
        return X()

class GraphCompKSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, mk = 4, C = 3.0, seed = 0, **kwargs):
        self._name = self.__class__.__name__
        self._logger = get_logger(self._name)
        self.mk = mk
        self.C = C ## only SVM cares about it
        self.idxs = None
        self.seed = seed
        self.N = -1
        self.K = None
        self.clf = SVC(kernel = "precomputed", C = self.C, 
                    class_weight = "balanced", random_state = self.seed, verbose = False)
        self._logger.debug("CONS: C={},mk={},seed={}, {}".format(self.C, self.mk, self.seed, self))
    
    ## be careful with it
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
            ## recursive things need to be set again 
            if parameter == "C":
                self.clf = SVC(kernel = "precomputed", C = self.C, 
                    class_weight = "balanced", random_state = self.seed, verbose = False)
        return self

    def __repr__(self):
        return "{}(eval={},C={},seed={},m={})".format(
            self._name, "ksvm", self.clf.C, self.seed, self.mk)
    
    def compute_protos(self, X, y):
        raise NotImplementedError("implement this")

    def fit(self, X, y, **fit_args):
        self._logger.debug("fit: data.shape {}, y.shape {}".format(X.shape, len(y)))
        self._logger.debug("fit: data[0] {}, y.shape {}".format(X[0], len(y)))
        self._logger.debug("fitting " + str(self))
        self.compute_protos(X, y)
        self.N = X.shape[0]
        if self.K is None: ## not none for methods that take vectors such as mmd-grad and kmeans
            self.K = X()[self.idxs, :][:, self.idxs]
        # self._logger.debug("\n" + str(self.K))
        cc = Counter(y[self.idxs])

        if self._name not in { "MMDCritic", "AllComp"}:
            assert set(cc.values()) == set({self.mk}), (self._name, dict(cc))
        
        try:
            self.clf.fit(self.K, y[self.idxs])
            assert self.clf.C == self.C, "problem applying C; {} != {}".format(self.C, self.clf.C)
            assert self.clf.kernel == "precomputed", "kernel should be precomputed"
        except Exception as ex:
            self._logger.error(str(ex) + str(self.clf))

        self._logger.debug("fitted {}".format(self))
        return self

    def predict(self, X):
        self._logger.debug("predict: data.shape {}".format(X.shape))
        self._logger.debug("predict: data[0] {}".format(X[0]))
        assert X.shape[1] == self.N, ("inconsistant data for precomputed kernels", X().shape, self.N)
        try:
            return self.clf.predict(X()[:, self.idxs])
        except Exception as ex:
            self._logger.warning(str(ex))
            return np.zeros(X.shape[0])
        
    def score(self, X, y, **bacc_argc):
        self._logger.debug("score: data.shape {}, y.shape {}".format(X.shape, len(y)))
        self._logger.debug("score: data[0] {}, y.shape {}".format(X[0], len(y)))
        self._logger.debug("scoring " + str(self))
        pred = self.predict(X)
        score = scorer(y, pred, **bacc_argc)
        return score

class NNComp(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, **kwargs):
        super().__init__(mk = mk, C = C, seed = 17, **kwargs)

    def __repr__(self):
        return "{}(C={},m={})".format(
            self._name, self.clf.C, self.mk)

    def compute_protos(self, X, y):
        W = X()
        idxss = []
        for c in np.array(sorted(set(y)), dtype = np.int8):
            mk = self.mk
            idxs = np.where(y == c)[0]
            nnsubm = FacilityLocation(W[idxs, :][:, idxs])
            S, _ = greedy_maximize(nnsubm, budget = mk)
            assert set(y[idxs[S]]) == {c}, (set(y[idxs[S]]), {c}) 
            idxss.append(idxs[S])
        self.idxs = np.array(idxss, dtype = np.uint32).flatten().tolist()
        return 

class PRComp(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, alpha = 0.85, **kwargs):
        self.alpha = alpha
        self._name = "PR"
        super().__init__(mk = mk, C = C, seed = 111, **kwargs)
    
    def __repr__(self):
        return "{}(C={},alpha={},m={})".format(
            self._name, self.clf.C, self.alpha, self.mk)
    
    def compute_protos(self, X, y):
        idxs = []
        for c in sorted(set(y)):
            idxs_c = np.where(y == c)[0]
            G = nx.from_numpy_matrix(X()[idxs_c, :][:, idxs_c])
            pr = nx.pagerank_scipy(G, alpha = self.alpha)
            res = nlargest(self.mk, pr.items(), key=itemgetter(1))
            idxs.append(idxs_c[list(map(itemgetter(0), res))] )
        self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()
        return

class CentralityComp(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, measure = "harmonic", **kwargs):
        self.measure = measure
        assert self.measure in {"harmonic", "betweenness", "closeness", "degree"}, \
                            "{}-centrality not supported".format(self.measure)
        self._name = self.measure
        super().__init__(mk = mk, C = C, seed = 12, **kwargs)
    
    def compute_protos(self, X, y):
        idxs = []
        fs = {"betweenness": nx.algorithms.centrality.betweenness_centrality,
            "closeness": nx.algorithms.centrality.closeness_centrality,
            "harmonic": nx.algorithms.centrality.harmonic_centrality,
            "degree": nx.algorithms.centrality.degree_centrality
        }
        for c in sorted(set(y)):
            idxs_c = np.where(y == c)[0]
            G = nx.from_numpy_matrix(X()[idxs_c, :][:, idxs_c])
            scores = fs[self.measure](G)
            res = nlargest(self.mk, scores.items(), key=itemgetter(1))
            idxs.append(idxs_c[list(map(itemgetter(0), res))] )
        self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()
        return

# class MMDCritic(GraphCompKSVM):
#     def __init__(self, mk = 4, C = 1.0, seed = 17, original_critic=True, **kwargs):
#         self._name = "MMDCritic"
#         self.original_critic = original_critic
#         super().__init__(mk = mk, C = C, seed = seed, **kwargs)
    
#     def __repr__(self):
#         return "{}(C={},m={})".format(
#             self._name, self.clf.C, self.mk)

#     def compute_protos(self, X, y):
        
#         clss = sorted(set(y))
#         V = np.arange(len(y)).tolist()
#         s = np.arange(len(y)).tolist()
        
#         ps = mmd.greedy_maximize(candidates = s, V = V, k = (self.mk * len(clss)) // 2, verbose = False, K = X)
#         s = [i for i in V if i not in ps]
#         reg_critic = (critic1 + logdet1) if self.original_critic else (critic2 + logdet2)
#         cs = reg_critic.greedy_maximize(V = V, candidates = s, k =  (self.mk * len(clss)) // 2, K = X, P = ps)
#         self.idxs = np.array(ps + cs)
        
#         return

class MMDGreedy(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, lambdaa = 0.21, diff = "diff", **kwargs):
        self.diff = diff
        self.lambdaa = lambdaa
        assert self.diff in {"diff", "div"}
        self._name = "MMD-{}-greedy".format(self.diff)
        super().__init__(mk = mk, C = C, seed = 13, **kwargs)
    
    def __repr__(self):
        return "{}(C={},lambda={},m={})".format(
            self._name, self.clf.C, self.lambdaa, self.mk)

    def compute_protos(self, X, y):
        W = X()
        idxs = []
        
        for c in np.array(sorted(set(y)), dtype = np.int8):
            mk = self.mk
            idxs_c = np.where(y == c)[0]
            # print(c, len(idxs_c), mk)
            mmd = Mmd_Comp(W, y, c, self.lambdaa, self.diff)
            candidates = np.zeros(len(W), dtype = np.int8)
            candidates[idxs_c] = 1
            S, _ = greedy_maximize(mmd, candidates, budget = mk)
            assert set(y[S]) == {c}
            idxs.append( S )
        self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()
        return


# class KmedoidsComp(GraphCompKSVM):
#     def __init__(self, mk = 4, C = 1.0, seed = 17, gamma = 0.1, **kwargs):
#         self.gamma = gamma
#         self._name = "kmedoids"
#         super().__init__(mk = mk, C = C, seed = seed, **kwargs)
    
#     def __repr__(self):
#         return "{}(C={},gamma={},m={})".format(
#             self._name, self.clf.C, self.gamma, self.mk)

#     def compute_protos(self, X, y):
#         clss = sorted(set(y))
#         idxs = []
#         for c, k in enumerate(clss):
#             idxs_c = np.where(y == k)[0]
#             kmpp_idxs = kmeanspp(X[idxs_c], self.mk, seed = self.seed)
#             self._logger.debug("kmedoids for {}, X.shape={}, init={}".format(k, X[idxs_c].shape, kmpp_idxs))
#             kmed = KMedoids(self.mk, init = kmpp_idxs)
#             kmed.fit(X[idxs_c], dist = False)
#             idxs.append(idxs_c[kmed.medoids].tolist())
#         self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()
#         self.K = GramMatrix.create(X[self.idxs], metric = "rbf", gamma = self.gamma)()
#         self.vecs_snapped = X[self.idxs]
#         return
    
#     def predict(self, X):
#         K = pairwise_kernels(X, self.vecs_snapped, metric = "rbf", gamma = self.gamma)
#         return self.clf.predict(K)

class RandomComp(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, seed = 19, **kwargs):
        self._name = "random"
        super().__init__(mk = mk, C = C, seed = seed, **kwargs)
    
    def __repr__(self):
        return "{}(C={},m={})".format(self._name, self.clf.C, self.mk)

    def compute_protos(self, X, y):
        clss = np.array(sorted(set(y)), dtype = np.int8)
        idxs = []
        for c, k in enumerate(clss):
            idxs_c = np.where(y == k)[0]
            idxs.append(idxs_c[:self.mk])
        self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()  
        # self.K = GramMatrix.create(X[self.idxs], metric = "rbf", gamma = self.gamma)()
        # self.vecs_snapped = X[self.idxs]
        return

class AllComp(GraphCompKSVM):
    def __init__(self, C = 3.0, seed = 19, **kwargs):
        self._name = "random"
        super().__init__(C = C, seed = seed, **kwargs)
    
    def __repr__(self):
        return "{}(C={},m=all)".format(self._name, self.clf.C)

    def compute_protos(self, X, y):
        self.idxs = np.arange(len(y))
        return

class LengthComp(GraphCompKSVM):
    def __init__(self, mk = 4, C = 3.0, seed = 19, gamma = 0.1, **kwargs):
        self.gamma = gamma
        self._name = "length"
        super().__init__(mk = mk, C = C, seed = seed, **kwargs)
    
    def __repr__(self):
        return "{}(C={},gamma={},m={})".format(
            self._name, self.clf.C, self.gamma, self.mk)

    def compute_protos(self, X, y):
        clss = np.array(sorted(set(y)), dtype = np.int8)
        idxs = []
        for c, k in enumerate(clss):
            idxs_c = np.where(y == k)[0]
            lengths = np.array((X[idxs_c, :]>0).sum(axis=1)).squeeze()#np.array(list(map(len, X[idxs_c])))
            # print(lengths)
            S = idxs_c[np.argpartition(lengths, -self.mk)[-self.mk:]].tolist()
            # print(S)
            assert set(y[S]) == {k}
            idxs.append(S)
        self.idxs = np.array(idxs, dtype = np.uint32).flatten().tolist()  
        # self.K = GramMatrix.create(X[self.idxs], metric = "rbf", gamma = self.gamma)()
        self.K = pairwise_kernels(X[self.idxs], metric = "cosine").squeeze()
        # print(y[self.idxs])
        # print(K)
        # self.K = np.exp(-self.gamma) * np.exp(K)
        self.vecs_snapped = X[self.idxs].todense()
        return
    
    def predict(self, X):
        K = pairwise_kernels(X, self.vecs_snapped, metric = "cosine").squeeze() #pairwise_kernels(X, self.vecs_snapped, metric = "rbf", gamma = self.gamma)
        return self.clf.predict(K)

def get_model(steps, **kwargs):
    steps_map = {
        "emb": ( PandasSelector("embeddings"), []),
        "ent": (PandasSelector("entities"), []),
        "tok": ( PandasSelector("tokens"), []),
        "bow": (BoW(weights = None), []),
        "idf-bow": (BoW(weights = "idf"), []),
        "llr-bow": (BoW(weights = "llr"), []),
        "tanh-bow": (BoW(weights = "tanh"), ["eta"]),
        "rbf": (RbfK(), ["gamma"]),
        "cos": (CosineK(), []),
        "linear": (LinearK(), []),
        "exp": (ExpK(), ["gamma"]),
        "thresh": (ThreshK(lower = kwargs.get("lower",None),
                        upper = kwargs.get("upper",None)), []),
        "nn": ( NNComp(**kwargs), ["C"]),
        "pr": ( PRComp(**kwargs), ["C", "alpha"] ),
        "degree": (CentralityComp(measure = "degree", **kwargs), ["C"] ) ,
        "betweenness": (CentralityComp(measure = "betweenness", **kwargs), ["C"] ) ,
        "closeness": (CentralityComp(measure = "closeness", **kwargs), ["C"]),
        "harmonic": (CentralityComp(measure = "harmonic", **kwargs), ["C"]),
        # "mmdc": (MMDCritic(**kwargs), ["C"]),
        "greedy-diff": (MMDGreedy(diff = "diff", **kwargs), ["C", "lambdaa"]),
        "greedy-div": (MMDGreedy(diff = "div", **kwargs), ["C", "lambdaa"]),
        # "kmeans": (KmeansComp(**kwargs), ["C", "gamma"]),
        # "kmedoids": (KmedoidsComp(**kwargs), ["C", "gamma"]),
        "random": (RandomComp(**kwargs), ["C"]),
        "length": (LengthComp(**kwargs), ["C"]),
        "all": (AllComp(**kwargs), ["C"]),
        "unwrap": (UnGramMatrix(), []),
        "svm": (SVC(kernel = "precomputed", class_weight = "balanced", verbose = 0), ["C"])
    }
    hyperparams = []
    for step in steps:
        assert step in steps_map.keys(), "%s not supported"%(step)
        hyperparams.extend( zip(steps_map[step][1], [step]*len(steps_map[step][1]) ))
    steps = [( s, steps_map[s][0] ) for s in steps]
    temp = [hp[0] for hp in hyperparams]
    assert len(set(temp)) == len(temp), "invalid seq of steps"
    return steps, hyperparams

def main():
    pass 

if __name__ == "__main__":
    pass