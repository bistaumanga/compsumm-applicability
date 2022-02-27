import json, sys, os
import pandas as pd
import numpy as np
from collections import Counter
from commons.utils import get_logger

from sklearn.model_selection import train_test_split
from commons.functions import bacc as scorer_func
from submodular.models2 import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from itertools import product, chain
import multiprocessing as mp
from commons.tokenizer import word_tokenize1
from upper_bound import filter_rows, tokenize, LABELS, COMPS, SPLITS, dates, N_SENTS, SPLITS, scorer, read_data, TOPIC_MAP, MONTHS_MAP
import h5py

PROJ_ROOT = os.environ.get("PROJ_ROOT","./")

HYPERPARAMS_GRID = {
    "lambdaa": [0.1, 0.2, 0.3, 0.4],
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0],
    "alpha": [0.8, 0.85, 0.9]
}

mks = [ 4, 8, 16, 32 ]
CV = 3

STEPS = [
    # ["tok", "idf-bow", "cos", "greedy-diff"],
    # ["tok", "idf-bow", "exp", "greedy-diff"],
    # ["tok", "idf-bow", "cos", "greedy-div"],
    # ["tok", "idf-bow", "exp", "greedy-div"],
    # ["tok", "bow", "cos", "greedy-diff"],
    # ["tok", "bow", "exp", "greedy-diff"], ### this one
    # ["tok", "bow", "cos", "greedy-div"],
    # ["tok", "bow", "exp", "greedy-div"],
    # ["tok", "idf-bow", "length"],
    ["tok", "idf-bow", "cos", "random"],
    ["emb", "cos", "random"],
    # ["tok", "idf-bow", "cos", "pr"],
    # ["tok", "bow", "cos", "pr"],
    # ["tok", "idf-bow", "cos", "nn"],
    # ["tok", "bow", "cos", "nn"],
    # ["emb", "exp", "greedy-diff"]
]

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logger = get_logger("MMD-MBFC")
# seed, TOPIC_MAP[topic], label1, label2, from_, to_, name, mk
# "timestamp": "2019-09-10T14:34:10+00:00"
def writer(queue, topic):
    path = "{}/cbc_ideology_{}.csv".format(PROJ_ROOT, topic)
    logger.info("writing output to %s"%path)
    fout = open(path, "w")
    fout.write("seed,topic,comp,month,method,k,val,test,summ1,summ2,hyps\n")
    
    # bias_counter = defaultdict(int)
    while True:
        msg = queue.get()
        if (msg == 'KILL'):
            break
        logger.info(msg)
        fout.write(msg + "\n")
        fout.flush()
    fout.close()
    logger.info("Closing file and Queue")

def run(df1, idxs_tr, idxs_te, y_tr, y_te, steps, mk, seed, topic, from_, to_, label1, label2, callback = lambda x: x):
    try:
        pipe, _hyperparams = get_model(steps, mk = mk)
        name = "_".join(steps)
        hyperparams = {}
        for hp in _hyperparams:
            hp_name, step = hp
            hyperparams["{}__{}".format(step, hp_name)] = HYPERPARAMS_GRID[hp_name]
            # if step == "greedy-div" and hp_name == "lambdaa":
            #     hyperparams["{}__{}".format(step, hp_name)] = HYPERPARAMS_GRID["lambda2"]
        print("{}, m={}, params={}".format(name, mk, ";".join(hyperparams.keys()) ))

        clf = GridSearchCV(Pipeline(pipe), hyperparams, 
                cv = CV, n_jobs = 6, pre_dispatch = 6, verbose = 0, scoring = scorer, 
                return_train_score = False, error_score = 0.5)
    
        clf.fit(df1.iloc[idxs_tr], y_tr)
        train_score, test_score = clf.score(df1.iloc[idxs_tr], y_tr), clf.score(df1.iloc[idxs_te], y_te)

        val_score = clf.best_score_
    
        # logger.info("{} => {}[{}]: val:{:.4g}, test:{:.4g}".format(topic, name, mk, val_score, test_score))

        S = clf.best_estimator_.steps[-1][1].idxs
        #### mistake, to fix
        summ_idxs = idxs_tr[S]
        assert set(df1.ideology.iloc[summ_idxs[:mk]]) == {label1} and set(df1.ideology.iloc[summ_idxs[mk:]]) == {label2}# ( set(df1.ideology.iloc[summ_idxs[:mk]]), label1, set(df1.ideology.iloc[summ_idxs[mk:]]), label2 )
        Summaries = df1.iloc[summ_idxs].id.values
        Summaries1 = Summaries[:mk].tolist()
        Summaries2 = Summaries[mk:].tolist()

        gamma, lambdaa = "", ""
        if "exp" in steps:
            gamma = clf.best_estimator_.steps[-2][1].gamma
        C = clf.best_estimator_.steps[-1][1].C
        if "greedy" in clf.best_estimator_.steps[-1][0]:
            lambdaa = clf.best_estimator_.steps[-1][1].lambdaa
        hyps = "g{}_l{}_C{}".format(gamma, lambdaa, C)

        msg = "{},{},{}_{},{},{},{},{:.4g},{:.4g},{},{},{}".format(
            seed, TOPIC_MAP[topic], 
            label1, label2, 
            MONTHS_MAP[from_[:7]], name, 
            mk, val_score, test_score,
            ";".join(Summaries1), ";".join(Summaries2), hyps
        )
        return callback(msg)
    except Exception as ex:
        msg = "{},{},{}_{},{},{},{}".format(
            seed, TOPIC_MAP[topic], label1, label2, 
            MONTHS_MAP[from_[:7]], name, mk
        )
        logger.error(str(ex))
        logger.error("failed " + msg)

def main():
    topic = sys.argv[1]
    N_JOBS = int(sys.argv[2])
    # FEATS = sys.argv[3] # or bow
    # assert FEATS in {"emb", "bow"}

    logger.info("comps:{}".format(COMPS))
    df = read_data(topic, extract_tokens = True )
    
    f = h5py.File('%s/cache/%s_v3.hdf5'%(PROJ_ROOT, topic),'r')
    ids_map = { str(id_):idx for (idx,id_) in enumerate(f["ids"][:] )}
    X_all = f["embs"][:]
    logger.info("X.shape:{}, |ids:{}, df:{}".format(X_all.shape, len(ids_map), len(df) ) )
    # logger.info(ids_map)
    assert X_all.shape[0] == len(ids_map) == len(df)
    f.close()

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(N_JOBS)
    q = pool.apply_async(writer, (queue, topic) )
    jobs = []

    for dr in dates:
        from_, to_ = dr
        df1 = df.query('date >= @from_ and date <= @to_').reset_index(drop=True)
            
        logger.info( "topic:{}, {}:{}, #articles:{}, {}".format(topic, from_, to_, len(df1), Counter(df1["ideology"]) ) )


        for (label1, label2) in COMPS:
            logger.debug("comparing %s,%s, %s-%s"%(label1, label2, from_, to_))
            idxs, y = filter_rows(df1, label1, label2)
            df2 = df1.iloc[idxs].reset_index(drop=True)
            
            idxs = np.array([ ids_map[str(id_)] for id_ in df2["id"].values ])
            X = X_all[idxs, :]
            norm = np.linalg.norm(X, axis = 1)
            X = (X.T / norm).T

            df2["embeddings"] = [x for x in X]

            for seed in SPLITS:
                idxs_tr, idxs_te, y_tr, y_te = train_test_split(np.arange(len(y)), y, stratify = y, random_state = seed)

                for steps, mk in product(STEPS, mks):    
                    job = pool.apply_async(run, (df2, idxs_tr, idxs_te, y_tr, y_te, steps, mk, seed, topic, from_, to_, label1, label2, queue.put))
                    jobs.append(job)
            
    for job in jobs:
        job.get()

    queue.put('KILL')
    pool.close()
    logger.info("Done")


if __name__ == "__main__":
    main()
