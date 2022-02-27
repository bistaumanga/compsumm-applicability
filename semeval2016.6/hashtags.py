import numpy as np
import pandas as pd
import nltk, math
import sys, json, operator
from nltk.tokenize import TweetTokenizer
from commons.functions import avg_f1 as scorer_func
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from commons.functions import each_f1
from itertools import product, chain
from collections import Counter
from commons.utils import get_logger
from submodular.llr import llr_compare
logger = get_logger("hashtags")

lemma = nltk.wordnet.WordNetLemmatizer()

tt = TweetTokenizer(preserve_case=False)

from nltk.corpus import stopwords
stops = set(stopwords.words('english'))
stops.update({"#semst", "...", "rt"})
ps = nltk.stem.porter.PorterStemmer()

def tokenize(t):
    tokens = tt.tokenize(t)
    tokens = [tok for tok in tokens if not ((len(tok) == 1 and not tok.isalpha()) or tok in stops)]
    tokens = [ (ps.stem(tok) if tok[0] != "#" else tok) for tok in tokens]
    return tokens

train = pd.read_csv("./StanceDataset/train.csv", encoding= 'unicode_escape', engine='python')
del train["Opinion Towards"]
del train["Sentiment"]
train["tokens"] = [ tokenize(t) for t in train["Tweet"].values ]
train["num_tokens"] = [len(tt) for tt in train["tokens"].values ]

for target in set(train["Target"]):
    fp = open("%s.txt"%target,"w")
    idxs = np.where(train["Target"] == target)[0]
    vec = TfidfVectorizer(
            tokenizer = lambda x: x, ## we already have list of entities
            lowercase = False,
            binary = False,
            use_idf = False,
            preprocessor = None,
            norm = None,
            min_df = 2,
            max_df = 0.5
    )
    M = vec.fit_transform(train.iloc[idxs]["tokens"])
    print("analysing %s, #terms=%d"%(target, M.shape[1]))
    scores = np.array(M.sum(axis=0)).squeeze()#vec.idf_
    # print(scores)
    vocab = [v for v,i in sorted(vec.vocabulary_.items(), key=operator.itemgetter(1))]
    scored = sorted(zip(vocab, scores), key=operator.itemgetter(1), reverse = True)
    idxs_pos = np.where(train.iloc[idxs]["Stance"].values == "FAVOR")[0]
    idxs_neg = np.where(train.iloc[idxs]["Stance"].values == "AGAINST")[0]
    pos = M[idxs_pos, :].sum(axis=0)
    neg = M[idxs_neg, :].sum(axis=0)
    pd, nd = dict(zip(vocab, np.array(pos).squeeze() )), dict(zip(vocab, np.array(neg).squeeze() ))

    y = np.zeros(M.shape[0])
    y[idxs_pos] = 1.0
    y[idxs_neg] = -1.0
    # pd = {k:v for k,v in pd.items() if v > 0}
    # nd = {k:v for k,v in nd.items() if v > 0}
    # srllr = llr_compare(pd, nd)
    # print(srllr)
    temp = np.array(1. / scores * M.T.dot(y)).squeeze()
    # print(temp)
    srllr = dict(zip(vocab,  temp) )
    for v, s in scored:
        fp.write("%s,%d,%d,%d,%.3g\n"%(v, s, pd[v],nd[v],srllr[v]
    ))

fp.close()