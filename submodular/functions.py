from submodular.base import Function
import numpy as np
import scipy as sp
from submodular import *
from profilehooks import profile
from scipy.linalg import lstsq, solve
from collections import Counter
from itertools import chain
from scipy.stats import describe
from commons.functions import kl_counters, softmax 

class FacilityLocation(Function):
    '''
        F(S+s) - F(S) for facility location
    '''
    def __init__(self, W, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._Fi = np.zeros(self._N, dtype = np.float)
        self._W = W
        
    def add2S(self, s):
        self._Fi = np.maximum(self._Fi, self._W[:, s]) ## set first time
        return super().add2S(s)

    def Fi(self, S = None): 
        return self._Fi if S is None else np.max(self._W[:, np.where(S == 1)[0]], axis = 1)

    def F(self, S = None, **kwargs):
        return self.Fi(S).sum()

    def marginal_gain(self, S = None, **kwargs):
        if (self._nS if S is None else np.sum(S)) == 0:
            return self._W.sum(axis = 0)
        FiS = self.Fi(S)
        gains = np.maximum(self._W.T, FiS) - FiS
        return gains.T.sum(axis = 0)

class SumCover(Function):
    '''
        F(S+s) - F(S) for  Sum(Sum(w_is))
        - is modular
    '''
    def __init__(self, W, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._r = W.sum(axis = 0)
        self._cache = 0.0
    
    def add2S(self, s):
        self._cache += self._r[s] ## set first time
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        return self._cache if S is None else np.dot(self._r, S)

    def marginal_gain(self, S = None, **kwargs):
        return self._r

class SumDiv(Function):
    '''
        F(S+s) - F(S) for Sum(w_ss')
        is submodular
    '''
    def __init__(self, W, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._cache_sum = 0.0
        self._cache_vec = np.zeros(self._N)
        self._W = W
    
    def add2S(self, s):
        self._cache_sum += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_sum += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)
    
    def F(self, S = None, **kwargs):
        ## mean to normalize, doesn't matter if sum is used
        return -1.0 * ( self._cache_sum if S is None else np.dot(np.dot(self._W, S), S))
        
    def marginal_gain(self, S = None, **kwargs):
        # idxs = np.where( (self._S if S is None else S) == 0 )[0]
        temp = self._W.diagonal().copy()#[idxs]
        temp += 2 * (self._cache_vec if S is None else np.dot(S, self._W))
        return -1.0 * temp

class Cut(Function):
    '''
        F(S+s) - F(S) for  cut
        = SumDiv + \lambda * SumDiv
        - is submodular if lambda >= 0
        - monotone if lambda in [0, 0.5]
        - generalization of MMR paper Goldstein et al 1998
    '''
    def __init__(self, W, lambdaa = 0.5, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._lambda = lambdaa
        assert self._lambda >= 0, "invalid lambdaa"
        self._cov = SumCover(W)
        self._div = SumDiv(W)
    @property
    def lambdaa(self): return self._lambda

    def __repr__(self):
        return str(self._cov) + str(self._div)

    def add2S(self, s):
        self._cov.add2S(s)
        self._div.add2S(s)
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        return self._cov(S) + self._lambda * self._div(S)

    def marginal_gain(self, S = None, **kwargs):
        return self._cov.marginal_gain(S) + self._lambda * self._div.marginal_gain(S)

class KL_raw(Function):
    """
        KL Divergence bases
        D: bag of semantic units array
    """
    def __init__(self, D, **kwargs):
        super().__init__(len(D))
        self._D = D
        self._D_counter = Counter(chain(*D))
        self._D_size = sum(self._D_counter.values())

        self._S_counter = Counter()
        self._S_size = 0

    def add2S(self, s):
        self._S_counter += Counter(self._D[s])
        self._S_size += len(self._D[s])
        return super().add2S(s)
    
    def F(self, S = None, **kwargs):
        if S is not None:
            S_counter = Counter(self._D[np.where( S == 1)[0]])
            S_size = sum(S_counter.values())
            return kl_counters(S_counter, self._D_counter, S_size, self._D_size)
        return kl_counters(self._S_counter, self._D_counter, self._S_size, self._D_size)
    
    def marginal_gain(self, S = None, **kwargs):
        msk = ((self._S if S is None else S) == 1)
        res = np.zeros(self._N)
        if S is not None:
            S_counter = Counter(self._D[np.where(msk)[0]])
            S_size = sum(S_counter.values()) 
        else:
            S_counter = self._S_counter
            S_size = self._S_size
        F_S = kl_counters(S_counter, self._D_counter, S_size, self._D_size)
        for s in np.where(~msk)[0]:
            F_Ss = kl_counters(
                S_counter + Counter(self._D[s]),
                self._D_counter,
                S_size + len(self._D[s]),
                self._D_size
            )
            res[s] = (F_Ss - F_S)
        return res

class AvgCover(Function):
    '''
        F(S+s) - F(S) for Avg(w_is)
        is submodular
    '''

    def __init__(self, W, f = None, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self.pV = 1./W.shape[0] * np.ones(W.shape[0], dtype = np.float)
        if f is not None:
            assert len(f) == W.shape[0]
            z = np.exp(f - np.max(f))
            self.pV = z / np.sum(z)
            print(describe(self.pV), describe(f))
        self._r = np.dot(W, self.pV)
        assert np.all(self._r > -1e8)
        self._cache = 0.0

    def add2S(self, s):
        self._cache += self._r[s] ## set first time
        return super().add2S(s)
    
    def F(self, S = None, **kwargs):
        ## mean to normalize, doesn't matter if sum is used
        sm = self._cache if S is None else np.dot(self._r, S)
        return sm / ( np.sum(S if S is not None else self._S))
    
    def marginal_gain(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        s1 = self._r.copy()#[np.where( (self._S if S is None else S) == 0 )[0] ]
        if lenS > 0:
            s1 -= self.F(S)
        return s1 / (lenS + 1.0 ) ## normalizer removed

class AvgDiv(Function):
    '''
        F(S+s) - F(S) for -Avg(w_is)
        is submodular
    '''

    def __init__(self, W, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._cache_sum = 0.0
        self._cache_vec = np.zeros(self._N)
        self._W = W
    
    def add2S(self, s):
        self._cache_sum += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_sum += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        if lenS > 0:
            temp = ( self._cache_sum if S is None else np.dot(np.dot(self._W, S), S))
            return -1.0 * temp / ( lenS * lenS)
        return 0.0
    
    def marginal_gain(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        s2 = - 1.0 * self._W.diagonal()
        if lenS > 0:
            s2 -= 2 * (self._cache_vec if S is None else np.dot(S, self._W))
            s2 -= (2.0 * lenS + 1.0) * self.F(S)
        return s2/( (lenS + 1.0) ** 2)

class MMD(Function):
    '''
        F(S+s) - F(S) for MMD (Kim et al NIPS 2016)
        = AvgDiv + 2 * AvgCover

    '''
    def __init__(self, W, f = None, **kwargs):
        super().__init__(len(W))
        self._cov = AvgCover(W, f)
        self._div = AvgDiv(W)
    
    def add2S(self, s):
        self._cov.add2S(s)
        self._div.add2S(s)
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        return 2 * self._cov(S) + self._div(S)

    def marginal_gain(self, S = None, **kwargs):
        return 2 * self._cov.marginal_gain(S) +  self._div.marginal_gain(S)

class LogDet(Function):
    '''
        F(S+s) - F(s) for  log-determinant K_SS (DPP paper 2013)
        - is submodular
        - monotone if smallest eigenvalue is atleast 1,
        - use lambda to control minimum eigenvalue
    '''
    def __init__(self, W, lambdaa = 1.0, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._lambda = lambdaa
        assert self._lambda >= 0 and self._lambda <= 1
        self._W = W + self._lambda * np.eye(self._N)
        self._S = np.zeros(self._N)

    def F(self, S = None, **kwargs):
        idxs = np.where( (self._S if S is None else S) == 1)[0]
        eigvals = sp.linalg.eigvalsh(self._W[idxs, :][:, idxs])
        eigvals[eigvals < EPS] = EPS
        return np.log(eigvals).sum()

    def marginal_gain(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        if lenS == 0:
            return self._W.diagonal()
        idxs = np.where( (self._S if S is None else S) == 1)[0]
        K_SS = self._W[idxs, :][:, idxs]
        K_Ss = self._W[idxs, :]
        k_ss = self._W.diagonal()
        try:
            term = solve(K_SS, K_Ss, assume_a = "pos")
        except:
            term = lstsq(K_SS, K_Ss)[0]
        det = np.abs(k_ss - (term * K_Ss).sum(axis = 0))
        det[det < EPS] = EPS
        return np.log(det)

class MMD_w(Function):
    '''
        F(S+s) - F(S) for MMD (Kim et al NIPS 2016)
        = AvgDiv + 2 * AvgCover

    '''
    def __init__(self, K, f, **kwargs):
        super().__init__(len(K))
        self._f = f#.detach().squeeze()
        # print(describe(self._f))
        self.pV = softmax(self._f)#F.softmax( self._f, dim = 0 )
        # print(describe(self.pV))
        # print(self.pV)
        self._K = K#
        self._r = np.dot(self._K, self.pV)
        self._o1 = np.dot(self.pV, self._r)
        self._C = 0.0 #np.max(self._f)
        assert self._K.shape == ( len(self._r), len(self._r) )

    def add2S(self, s):
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        mmd = self._o1
        if lenS > 0:
            # print(S, self._K.shape )
            qS = softmax(self._f[S_].squeeze())
            K_SS = self._K[S_, :][:, S_].squeeze()
            mmd -= 2.0 * np.dot(self._r[S_], qS)
            mmd += np.dot(qS, np.dot( K_SS, qS ))
        return -mmd
    
    def marginal_gain(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        e_fS = 0.0
        term1, term3, term4 = 0.0, 0.0, 0.0
        e_fsk = np.exp(self._f - self._C)
        if lenS > 0:
            qS = softmax(self._f[S_])
            e_fS = np.sum( np.exp(self._f[S] - self._C) )
            term1 = np.dot(self._r[S_], qS) * (- e_fsk / (e_fsk + e_fS) )
            
            term3 = np.dot(qS, np.dot(self._K[S_, :][:, S_].squeeze(), qS)) * \
                            (e_fS ** 2 - (e_fS + e_fsk) ** 2) / (e_fS + e_fsk ) ** 2
            tmp = (e_fsk / (e_fS + e_fsk)) * 1./ (1. + e_fsk / e_fS).squeeze()
            term4 = np.dot(self._K[:, S_], qS).squeeze() * tmp 
            # print(term1.shape, term3.shape, term4.shape)
        # print(type(self._r), type(e_fsk))
        term2 = self._r * e_fsk / ( e_fsk + e_fS )
        term5 = ( e_fsk / (e_fS + e_fsk) ) ** 2 * np.diagonal(self._K)
        # print(term2.shape, term5.shape)
        res = 2. * term1 + 2. * term2 - term3 - 2. * term4 #- term5
        return res