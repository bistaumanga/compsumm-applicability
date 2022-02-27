import json, sys, time,os
import pandas as pd
import numpy as np
from collections import Counter
from commons.utils import get_logger
from timeit import default_timer as timer
import nltk
import h5py

from sklearn.feature_extraction.text import CountVectorizer as BoW
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from commons.functions import bacc as scorer_func
from submodular.models2 import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from itertools import product, chain
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
import multiprocessing as mp
from commons.tokenizer import word_tokenize1

PROJ_ROOT = os.environ.get("PROJ_ROOT","./")
CV = 5
scorer = make_scorer(scorer_func)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
N_SENTS = 3 #or "None" for all sents
if type(N_SENTS) is int:
    assert N_SENTS >= 1
else:
    N_SENTS = None

logger = get_logger("SVM-MBFC")

SPLITS = (0,1,2,3,4,5,6,7,8,9)
dates = [
    ("2019-01-01", "2019-01-31"),
    ("2019-02-01", "2019-02-28"),
    ("2019-03-01", "2019-03-31"),
    ("2019-04-01", "2019-04-30"),
    ("2019-05-01", "2019-05-31"),
    ("2019-06-01", "2019-06-30"),
    ("2019-07-01", "2019-07-31"),
    ("2019-08-01", "2019-08-31"),
    ("2019-09-01", "2019-09-30")
]
LABELS = ["ER", "R", "RC", "C", "LC", "L"]
COMPS = [(a, b) for (a, b) in product(LABELS, LABELS) if a < b]

MONTHS = ["2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09"]
MONTHS_MAP = dict(zip(MONTHS, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]))

TOPIC_MAP = {
    "chr01": "LGBT",
    "auspol": "AusPol",
    "climatechange": "ClimateChange",
    "guncontrol": "GunControl",
    "cbp01": "IllegalImmi",
    "cbp02": "AsylumSeekers"
}


def filter_rows(df, label1, label2):
    idx1 = np.where(df.ideology == label1 )[0]
    idx2 = np.where(df.ideology == label2 )[0]
    # assert len(idx1) + len(idx2) == len(df), (len(idx1), len(idx2), len(df))
    idxs = np.array(idx1.tolist() + idx2.tolist())
    labels = np.array([-1] * len(idx1) + [1] * len(idx2))
    return idxs, labels

def writer(queue, topic, feats, method):
    fout = open("%s/%s_ideology_%s_%s_nonorm.csv"%(PROJ_ROOT, method, topic, feats), "w")
    fout.write("topic,comp,seed,month,num_tr1,num_tr2,num_te1,num_te2,val,test,num_feats,time\n")
    # bias_counter = defaultdict(int)
    while True:
        msg = queue.get()
        if (msg == 'KILL'):
            break
        fout.write(msg + "\n")
        fout.flush()
    fout.close()
    logger.info("Closing file and Queue")

def run(X_tr, X_te, y_tr, y_te, msg, dual, loss, callback = lambda x: x):
    start = timer()
    
    ## Kernel SVC -- probably take day+, requires much memory
    ## see sklearn's doc for SVC for LinearSVC
    # hyperparams = {"C": [ 1e+3, 1e+2, 1e+1, 1.0, 0.1 ],  "gamma" : [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0]}
    # clf = SVC(kernel = "rbf", class_weight = 'balanced', random_state = 1)
    
    ## linear SVC -- super fast
    if loss in {"hinge", "squared_hinge"}:
        hyperparams = {"C": [ 1e+3, 1e+2, 1e+1, 1.0, 0.1 ]}
        clf = LinearSVC(loss = loss, dual = dual, class_weight = 'balanced', random_state = 1, verbose = 0)
    ## Logreg using SGD -- fast
    elif loss == "log":
        hyperparams = {"alpha": [ 1e-5, 1e-4, 1e-3, 1e-2, 1e-1 ]}
        clf = SGDClassifier( loss = loss, penalty = "l2", random_state = 1, class_weight = 'balanced', verbose = 0, tol = 1e-5, max_iter=1000, eta0 = 0.01, learning_rate = "adaptive")
    
    
    clf = GridSearchCV( clf, 
        hyperparams, cv = CV, n_jobs = 5, 
        pre_dispatch = 5, verbose = 0, 
        scoring = scorer, return_train_score = False,
        error_score = 0.5)
    
    clf.fit(X_tr, y_tr)
    test_score = clf.score(X_te, y_te)
    val_score = clf.best_score_
    logger.info("{} => val:{:.4g}, test:{:.4g}".format(msg, val_score, test_score))
    time_run = timer() - start
    return callback(msg + ",{:.4g},{:.4g},{},{:.2f}".format(val_score, test_score, X_tr.shape[1],time_run))

def tokenize(row):
    title_toks = word_tokenize1(row["title"])
    des_toks = list(chain(*[word_tokenize1(s) for s in row["sents"] ] ))
    return title_toks + des_toks

def read_data(topic, extract_tokens = True):
    df = pd.read_json("{}/dataset/filtered_{}.json".format(PROJ_ROOT, topic), orient = "records", lines = True)
    df.drop([ "domain", "url" ], inplace = True, axis = 1)

    logger.info("data read, #rows:{}".format(len(df)))
    logger.info("Distribution: {}".format(Counter(df["ideology"])))
    
    df = df.query("ideology in @LABELS").reset_index()
    df["sents"] = [ s[:N_SENTS] for s in df["sents"] ]

    if extract_tokens:
        df["tokens"] = df.apply(tokenize, axis = 1)
        df["n_toks"] = df.apply(lambda row: len(row["tokens"]), axis = 1 )
        # df.drop([ "title", "sents" ], inplace = True, axis = 1)
    return df


def main():
    topic = sys.argv[1]
    FEATS = sys.argv[2] # or bow
    assert FEATS in {"emb", "bow"}, FEATS
    LOSS = sys.argv[3]
    assert LOSS in {"hinge", "squared_hinge", "log"}, LOSS
    N_JOBS = int(sys.argv[4])

    logger.info("comps:{}".format(COMPS))
    df = read_data(topic, FEATS == "bow" )
    
    if FEATS == "emb":
        f = h5py.File('cache/%s_v3.hdf5'%topic,'r')
        ids_map = { str(id_):idx for (idx,id_) in enumerate(f["ids"][:] )}
        X_all = f["embs"][:]
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # X_all = scaler.fit_transform(X_all)

        logger.info("X.shape:{}, |ids:{}, df:{}".format(X_all.shape, len(ids_map), len(df) ) )
        # logger.info(ids_map)
        assert X_all.shape[0] == len(ids_map) == len(df), (len(ids_map), len(df))
        f.close()
    
    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(N_JOBS, maxtasksperchild = N_JOBS)
    q = pool.apply_async(writer, (queue, topic, FEATS, LOSS) )
    jobs = []

    for dr in dates:
        from_, to_ = dr
        df1 = df.query('date >= @from_ and date <= @to_').reset_index(drop=True)
            
        logger.info( "topic:{}, {}:{}, #articles:{}, {}".format(topic, from_, to_, len(df1), Counter(df1["ideology"]) ) )

        for (label1, label2) in COMPS:
            idxs, y = filter_rows(df1, label1, label2)
            df2 = df1.iloc[idxs].reset_index(drop=True)
            start = time.time()
            if FEATS == "bow":
                ## prepare bow
                bow = TfidfVectorizer( tokenizer = lambda x: x, lowercase = False, min_df = 2, preprocessor = None, use_idf = False, norm = "l2", max_df = 0.2, max_features = 2000 )
                X = bow.fit_transform(df2["tokens"].values)    
                X = np.array(X.todense()).squeeze()

            else:
                idxs = np.array([ ids_map[str(id_)] for id_ in df2["id"].values ])
                X = X_all[idxs, :] # / 4.0 ## because the embeddings are sum of title + 3 sents
                X = (X.T / np.linalg.norm(X, axis = 1)).T

            logger.info("{}-{}: X,shape = {} in {} secs".format(label1, label2, X.shape, time.time() - start ))
            assert X.shape[0] == len(df2)

            for seed in SPLITS:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify = y, random_state = seed)
                if FEATS == "bow":
                    feats_idxs = np.where(X_tr.sum(axis=0) > 0)[0]
                    X_tr = X_tr[:, feats_idxs]
                    X_te = X_te[:, feats_idxs]

                cc_tr = Counter(y_tr)
                cc_te = Counter(y_te)
                dual = X_tr.shape[0] > X_tr.shape[1]

                msg = "{},{}_{},{},{},{},{},{},{}".format(
                    TOPIC_MAP[topic], label1, label2, seed, MONTHS_MAP[from_[:7]],
                    cc_tr[-1], cc_tr[1], cc_te[-1], cc_te[1] 
                )
                
                job = pool.apply_async(run, (X_tr, X_te, y_tr, y_te, msg, dual, LOSS, queue.put))
                jobs.append(job)
    pool.close()      
    for job in jobs:
        job.get()

    queue.put('KILL')
    logger.info("Done")

if __name__ == "__main__":
    main()