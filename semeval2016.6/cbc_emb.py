import json, sys, time, os, glob
import pandas as pd
import numpy as np
from collections import Counter
from commons.utils import get_logger
from timeit import default_timer as timer
import nltk, h5py, math

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from itertools import product, chain
from sklearn.svm import SVC, LinearSVC
import multiprocessing as mp
from data import TweetStanceDataset, STANCE_MAP_INV, STANCE_MAP
from upper_bound_emb import TARGETs, write_preds, eval_cm
# from model import *
from nltk.tokenize import TweetTokenizer
import torch
from torch.optim import LBFGS, Adam, SGD
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from submodular.models2 import *
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

np.random.seed(19)

from commons.functions import bacc as scorer_func
ERROR_SCORE = 1./3
# def scorer_func(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     f1 = eval_cm(cm)
#     return np.mean(f1[1:])
# ERROR_SCORE = 0.0

logger = get_logger("SemEval-CBC")

HYPERPARAMS_GRID = {
    "lambdaa": [0.1, 0.2, 0.3, 0.4],
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0],
    "alpha": [0.8, 0.85, 0.9]
}

mks = [5, 10] #8, 16, 24, 32]
CV = 5
N_JOBS = int(sys.argv[1])
scorer = make_scorer(scorer_func)
STEPS = [
    # ["tok", "idf-bow", "cos", "greedy-diff"],
    # ["tok", "idf-bow", "exp", "greedy-diff"],##
    # ["tok", "idf-bow", "cos", "greedy-div"],
    # ["tok", "idf-bow", "exp", "greedy-div"],
    # ["tok", "bow", "cos", "greedy-diff"],
    ["tok", "bow", "exp", "greedy-diff"], ##
    # ["tok", "bow", "cos", "greedy-div"],
    # ["tok", "bow", "exp", "greedy-div"],
    # ["tok", "idf-bow", "length"],
    # ["tok", "idf-bow", "cos", "random"],
    # ["tok", "idf-bow", "cos", "pr"],
    # ["tok", "bow", "cos", "pr"],
    # ["tok", "idf-bow", "cos", "nn"], ##
    ["tok", "bow", "cos", "nn"], ##
    ["emb", "rbf", "greedy-diff"] ##
]
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

VERBOSE = False

def main():
    # outfile = open("res_cbc.csv", "w+")
    # outfile.write("target,method,k,val,test,L1,L2,f1.Against,f1.For\n")
    summaries_file = open("summaries_bacc.json", "w+")
    datasets = {}
    # fp = open("upper_bound_emb.csv", "w")
    # fp.write("target,refit,oversample,loss,category,prec,recall,count\n")
    for target in TARGETs:
        try:
            train_dataset = TweetStanceDataset.load("cache", "train", target)
            logger.info(train_dataset)
        except:
            logger.info("caching train %s ..."%target)
            train_dataset = TweetStanceDataset("./StanceDataset/%s.csv"%"train", target = target)
            logger.info(train_dataset)
            train_dataset.save("cache", "train")
            logger.info("done caching train %s !"%target)
        try:
            test_dataset = TweetStanceDataset.load("cache", "test", target)
            logger.info(test_dataset)
        except:
            logger.info("caching test %s ..."%target)
            test_dataset = TweetStanceDataset("./StanceDataset/%s.csv"%"test", target = target)
            logger.info(test_dataset)
            test_dataset.save("cache", "test")
            logger.info("done caching test %s !"%target)

        ## CE loss sombines softmax and NLL
        datasets[target] = (train_dataset, test_dataset)
    
    for f in glob.glob("predictions/predsA_cbc_bacc_*.txt"):
        os.remove(f)
    # write_gold(datasets)
    
    for target, dataset in datasets.items():
        train_dataset, test_dataset = dataset
        df_train = train_dataset.df
        
        logger.info("tokenizing " + target )
        df_train["tokens"] = [ tokenize(t) for t in df_train["Tweet"].values ]
        df_train["num_tokens"] = [len(tt) for tt in df_train["tokens"].values ]
        X = train_dataset.emb.numpy()
        # X /= np.linalg.norm(X, axis = 0)
        assert X.shape[0] == len(df_train)
        df_train["embeddings"] = X.tolist()
        # print(train_dataset.emb.numpy().shape)
    
        df_test = test_dataset.df
        df_test["tokens"] = [ tokenize(t) for t in df_test["Tweet"].values ]
        df_test["num_tokens"] = [len(tt) for tt in df_test["tokens"].values ]
        
        X = test_dataset.emb.numpy()
        # X /= np.linalg.norm(X, axis = 0)
        assert X.shape[0] == len(df_test)
        df_test["embeddings"] = X.tolist()
        
        logger.info("Topic: {}, #tweets:{} ".format(target, dict(Counter(df_train.Stance.values))))
        
        for steps, mk in product(STEPS, mks):    
            if target == "Climate Change is a Real Concern" and mk > 15:
                continue
            pipe, _hyperparams = get_model(steps, mk = mk)
            name = "_".join(steps)
            hyperparams = {}
            for hp in _hyperparams:
                hp_name, step = hp
                hyperparams["{}__{}".format(step, hp_name)] = HYPERPARAMS_GRID[hp_name]
            logger.info("  - Summarizing {}, m={}, params={}".format(name, mk, ";".join(hyperparams.keys()) ))

            Summaries = {}
            S_all = []
            
            df_train["y"] = [STANCE_MAP_INV.index(s) for s in df_train.Stance]
            df_test["y"] = [STANCE_MAP_INV.index(s) for s in df_test.Stance]
                
            clf = GridSearchCV(Pipeline(pipe), hyperparams, 
                        cv = CV, n_jobs = N_JOBS, pre_dispatch = N_JOBS, verbose = 0, scoring = scorer, 
                        return_train_score = False, error_score = ERROR_SCORE)
            
            clf.fit(df_train, df_train.y.values)
            train_score, test_score = clf.score(df_train, df_train.y.values), clf.score(df_test, df_test.y.values)
            val_score = clf.best_score_
            preds = clf.predict(df_test)
            assert len(preds) == len(df_test)
            write_preds( "predictions/predsA_cbc_bacc_{}_{}.txt".format(name, mk), preds, test_dataset )
            logger.info("    Tr: {:.4g}, Val:{:.4g}, Test:{:.4g}".format(train_score, val_score, test_score))
            S = clf.best_estimator_.steps[-1][1].idxs
            logger.debug(Counter(df_train.y.values[S]))
            assert list(Counter(df_train.y.values[S]).values()) == [mk] * len(STANCE_MAP_INV) 
            Summaries = dict(zip(STANCE_MAP_INV, [ a.tolist() for a in np.array_split( df_train.iloc[S].Tweet.values, len(STANCE_MAP_INV) )]  ) )
            if mk <= 5:
                Summaries["name"] = name
                Summaries["mk"] = mk
                Summaries["target"] = target
                # print(Summaries)
                summaries_file.write( json.dumps(Summaries) + "\n" )
                # S_all.extend(S)       
    summaries_file.close()

if __name__ == "__main__":
    main()