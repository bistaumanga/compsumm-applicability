from submodular.functions import *
from submodular.feature_based_functions import *
from submodular.base import *
from commons.functions import kl_counters, softmax

class Mmd_Comp(Function):
    '''
        F(S+s) - F(S) for MMD^2 comparative
        is submodular
    '''

    def __init__(self, W, y, g, lambdaa, diff = 'diff', **kwargs):
        assert diff in {'diff', 'div'}
        super().__init__(len(W))
        self._W = W
        
        assert len(self._W) == len(y)
        self._lambdaa = lambdaa
        assert self._lambdaa >= 0.0 and self._lambdaa <= 1.0
        self._d = 1.0 if diff == 'div' else (1 - self._lambdaa)
        f = self.f_hard if len(set(y)) == 2 else self.f_soft
        self._Pg, self._fg = f(y, g)
        self._r = np.mean(W * self._fg, axis = 1) ##
        assert np.all(self._r > -1e8)
        self._cache_cov = 0.0
        self._cache_div = 0.0
        self._cache_vec = np.zeros(self._N)
    
    def add2S(self, s):
        self._cache_cov += self._r[s] ## set first time
        self._cache_div += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_div += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)

    def f_(self, pgx, pg):
        return pgx/pg - self._lambdaa * ( 1-pgx) / (1.0- pg)

    def f_soft(self, y, g, **kwargs ):
        pgx = y*g/2.0 + 0.5
        pg = np.sum(pgx)/len(y)
        return pg, self.f_(pgx, pg)
    
    def f_hard(self, y, g, **kwargs):
        pgx = np.array(y == g, dtype = np.int)
        assert np.sum(pgx) > 0, "check if you passed hard labels"
        pg = np.sum(pgx)/len(y)
        return pg, self.f_(pgx, pg)	
    
    def F(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        cov = self._cache_cov if S is None else np.dot(self._r, S)
        div = 0.0
        if lenS > 0:
            div = ( self._cache_div if S is None else np.dot(np.dot(self._W, S), S))
        return 1. / lenS * ( 2.0 * cov - self._d * div / lenS)

    def marginal_gain(self, S = None, **kwargs):
        lenS = (self._nS if S is None else np.sum(S)) 
        cov = self._r.copy()
        if lenS > 0:
            cov -=  ( self._cache_cov if S is None else np.dot(self._r, S) ) / lenS
        div = - 1.0 * self._W.diagonal()
        if lenS > 0:
            div -= 2 * (self._cache_vec if S is None else np.dot(S, self._W))
            div += (2.0 * lenS + 1.0) / (lenS ** 2) * ( self._cache_div if S is None else np.dot(np.dot(self._W, S), S))
        return 1.0 / (lenS + 1.0) * (2.0 * cov + self._d * div / (lenS + 1.0))

class MMD_Comp_w(Function):
    '''
        F(S+s) - F(S) for MMD (Kim et al NIPS 2016)
        = AvgDiv + 2 * AvgCover
        = selects from VB

    '''
    def __init__(self, KB, KA, KAB, fB, fA, lambdaa = 0.2, diff = 1, **kwargs):
        super().__init__(len(KB))
        assert diff in {0, 1}
        self._fB = fB#.detach().squeeze()
        self._fA = fA#.detach().squeeze()
        # print(describe(self._fB))
        self.pVB = softmax(self._fB) #F.softmax( self._fB, dim = 0 )
        self.pVA = softmax(self._fA) #F.softmax( self._fA, dim = 0 )
        # print("pB", describe(self.pVB))
        # print("pA", describe(self.pVA))
        
        # print(self.pV)
        self._KB = KB
        self._KA = KA
        self._KAB = KAB
        assert self._KB.shape == ( len(self._fB), len(self._fB) ), (self._KB.shape, len(self._fB), len(self._fB) )
        assert self._KA.shape == ( len(self._fA), len(self._fA) ), (self._KA.shape, len(self._fA), len(self._fA) )
        assert self._KAB.shape == ( len(self._fA), len(self._fB) ), (self._KAB.shape, len(self._fA), len(self._fB) )
        print((self._KB.shape, self._KA.shape, self._KAB.shape, 
                self._fA.shape, self._fB.shape))

        self._rB = np.dot(self._KB, self.pVB)
        self._o1 = np.dot(self.pVB, self._rB)
        self._rA = np.dot(self._KAB.T, self.pVA)
        self._c1 = np.dot(self.pVA, np.dot(self._KA, self.pVA))
        self._CB = np.max(self._fB)
        self._CA = np.max(self._fA)
        assert len(self._rA) == len(self._rB)
        self.lambdaa = lambdaa
        assert self.lambdaa >= 0.0 and self.lambdaa < 0.9
        self.diff = diff

    def add2S(self, s):
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        mmd = self._o1 - self.lambdaa * self._c1
        if lenS > 0:
            # print(S, self._K.shape )
            qS = softmax(self._f[S_], dim = 0)
            K_SS = self._K[S_, :][:, S_].squeeze()
            mmd -= 2.0 * np.dot(self._rB[S_], qS)
            mmd += 2.0 * self.lambdaa * np.dot(self._rA[S_], qS)
            mmd += ( 1 - self.lambdaa * self.diff ) * np.dot(qS, np.dot( K_SS, qS ))
        return -mmd
    
    def marginal_gain(self, S = None, **kwargs):
        S_ = self._S if S is None else copy(S)
        lenS = (self._nS if S is None else np.sum(S))
        S_ = np.where(S_ == 1)[0]
        e_fSB, e_fSA = 0.0, 0.0
        term1B, term1A, term3B, term4B = 0.0, 0.0, 0.0, 0.0
        e_fskB = np.exp(self._fB - self._CB)
        
        if lenS > 0:
            qS = softmax(self._fB[S_])
            
            e_fSB = np.sum( np.exp(self._fB[S] - self._CB) )

            term1B = np.dot(self._rB[S_], qS) * (- e_fskB / (e_fskB + e_fSB) )
            term1A = np.dot(self._rA[S_], qS) * (- e_fskB / (e_fskB + e_fSB) )
            
            term3B = np.dot(qS, np.dot(self._KB[S_, :][:, S_].squeeze(), qS)) * \
                            (e_fSB ** 2 - (e_fSB + e_fskB) ** 2) / (e_fSB + e_fskB ) ** 2

            tmp = (e_fskB / (e_fSB + e_fskB)) * 1./ (1. + e_fskB / e_fSB).squeeze()
            term4B = np.dot(self._KB[:, S_], qS).squeeze() * tmp 
            # print(term1.shape, term3.shape, term4.shape)
        # print(type(self._r), type(e_fsk))
        term2B = self._rB * e_fskB / ( e_fskB + e_fSB )
        term2A = self._rA * e_fskB / ( e_fskB + e_fSB )
        term5B = ( e_fskB / (e_fSB + e_fskB) ) ** 2 * np.diagonal(self._KB)
        # print(term2.shape, term5.shape)
        res = 2. * ( term1B + term2B  ) 
        res -= (1.0 - self.lambdaa * self.diff) * ( term3B + 2. * term4B )
        res +=  term5B 
        res -= 2 * self.lambdaa * (term1A + term2A)
        return res

# class KLComp(Function):
#     """
#         KL Divergence bases
#         D: bag of semantic units
#     """
#     def __init__(self, D, **kwargs):
#         super().__init__(len(D))
#         self._D = D
#         self._D_counter = Counter(chain(*D))
#         self._D_size = sum(self._D_counter.values())

#         self._S_counter = Counter()
#         self._S_size = 0

#     def add2S(self, s):
#         self._S_counter += Counter(self._D[s])
#         self._S_size += len(self._D[s])
#         return super().add2S(s)
    
#     def F(self, S = None, **kwargs):
#         if S is not None:
#             S_counter = Counter(self._D[np.where( S == 1)[0]])
#             S_size = sum(S_counter.values())
#             return kl_counters(S_counter, self._D_counter, S_size, self._D_size)
#         return kl_counters(self._S_counter, self._D_counter, self._S_size, self._D_size)
    
#     def marginal_gain(self, S = None, **kwargs):
#         msk = ((self._S if S is None else S) == 1)
#         res = np.zeros(self._N)
#         if S is not None:
#             S_counter = Counter(self._D[np.where(msk)[0]])
#             S_size = sum(S_counter.values()) 
#         else:
#             S_counter = self._S_counter
#             S_size = self._S_size
#         F_S = kl_counters(S_counter, self._D_counter, S_size, self._D_size)
#         for s in np.where(~msk)[0]:
#             F_Ss = kl_counters(
#                 S_counter + Counter(self._D[s]),
#                 self._D_counter,
#                 S_size + len(self._D[s]),
#                 self._D_size
#             )
#             res[ix] = (F_Ss - F_S)
#         return res

class Li3Comp(Function):
    def __init__(self, W, y, g, lambdaa1 = 1.0, lambdaa2 = 1.0, **kwargs):
        super().__init__(len(W))
        assert W.shape == (self._N, self._N)
        self._W = W
        assert len(y) == self._N
        self._lambdas = [lambdaa1, lambdaa2]
        self._g = g
        self._y = y
        self._fg = (self._y == self._g)*np.ones(self._N)-self._lambdas[1]*(y != g)*np.ones(self._N)
        self._r = (W * self._fg).sum(axis = 1)
        assert np.all(self._r > - 1e8), self._r[np.where(self._r < - 1e8)[0]]
        self._cache_sum = 0.0
        self._cache_vec = np.zeros(self._N)
        self._cache = 0.0

    def add2S(self, s):
        self._cache += self._r[s] ## set first time
        self._cache_sum += 2.0 * np.dot(self._W[s, :], self._S) ## set first time
        self._cache_sum += self._W[s, s]
        self._cache_vec += self._W[:, s]
        return super().add2S(s)

    def F(self, S = None, **kwargs):
        cov = self._cache if S is None else np.dot(self._r, S)
        div = self._cache_sum if S is None else np.dot(np.dot(self._W, S), S)
        return cov - self._lambdas[0] * div

    def marginal_gain(self, S = None, **kwargs):
        res = -1e6 * np.ones(self._N)
        idxs = np.where(np.logical_and((self._S if S is None else S) == 0, self._y == self._g))[0]
        # idxs = np.where((self._S if S is None else S) == 0)[0]
        cov = self._r[idxs].copy()
        div = self._W.diagonal()[idxs]
        div += 2 * (self._cache_vec[idxs] if S is None else np.dot(S, self._W[:, idxs]))
        res[idxs] = cov - self._lambdas[0] * div
        return res
    
    def curvature(self, **kwargs):
        c = np.zeros(self._N, dtype = np.float32)
        V_s = np.ones(self._N, dtype = np.bool) * (self._y == self._g)
        idxs = np.where(self._y == self._g )[0]
        for s in idxs:
            V_s[s] = False
            val = 1.0 * self.marginal_gain(S = V_s)[0] / self( np.logical_not(V_s ))
            c[s] = val
            V_s[s] = True
        return 1 - np.min(c)
