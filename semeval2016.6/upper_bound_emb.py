import json, sys, time, os, glob
import pandas as pd
import numpy as np
from collections import Counter
from commons.utils import get_logger
from timeit import default_timer as timer
import nltk, h5py, math

from sklearn.model_selection import train_test_split
from commons.functions import bacc as scorer_func
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer
from itertools import product, chain
from sklearn.svm import SVC, LinearSVC
import multiprocessing as mp
from data import TweetStanceDataset, STANCE_MAP_INV, STANCE_MAP
from model import *
import torch
from torch.optim import LBFGS, Adam, SGD
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
# import torch.optim.lr_scheduler.StepLR
from sklearn.metrics import confusion_matrix
np.random.seed(19)

logger = get_logger("SemEval-UB")
TARGETs = ["Atheism",
    "Climate Change is a Real Concern",
    "Feminist Movement",
    "Hillary Clinton",
    "Legalization of Abortion"
]
MAX_EPOCHS = 200
LR, ALPHA = 1e-2, 1e-4
STORE_PATH = "./logs/"
BATCH_SIZE = 32
VERBOSE_INTERVAL = 20
EARLY_DELTA = 1e-7
EARLY_PATIENCE = 20
GPU_MODE = False
VERBOSE = False
LR_STEPS = 20

### one step over a mini batch
def step(model, data, idxs, loss, confusion_mat = False):
    loss_accum = torch.DoubleTensor([0.0])
    
    if GPU_MODE:
        loss_accum = loss_accum.cuda() 

    tweet, target, label, pooled_emb, ctx_emb = data[idxs]
    label = label.float().argmax(1)
    ## forward pass
    # norm = pooled_emb.norm(p=2, dim=1, keepdim=True)
    # pooled_emb_norm = pooled_emb.div(norm.expand_as(pooled_emb))
    # X /= np.linalg.norm(X, axis = 0)
    y_out = model.forward( pooled_emb )
    loss_val = loss(y_out, label )
    loss_accum += loss_val
    y_pred = y_out.argmax(1, keepdim=True)

    cm = None

    ### saving confusion matrix
    if confusion_mat:
        cm = confusion_matrix(label, y_pred)
        logger.debug(cm)
    loss_accum = loss_accum / len(idxs)
    return loss_accum, cm, y_pred

def train(model1, loss, train_dataset, train_idxs, val_idxs = None, NUM_EPOCHS = MAX_EPOCHS, oversample = True, weighted_loss = True):
    optimizer = Adam( model1.parameters(), lr = LR, weight_decay = ALPHA )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 10, factor = 0.1)
    # optimizer = SGD(model1.parameters(), lr=LR, momentum = 0.9, weight_decay = ALPHA)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEPS, gamma=0.1)
    # torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    writer = SummaryWriter('{}/{}{}_os{}wl{}_{}'.format(
        STORE_PATH, model1.name, 
        "_refit_" if val_idxs is None else "",
        int(oversample), int(weighted_loss) ,
        train_dataset.target.replace(" ", "_")
    ))
    epoch = 1
    ## setting up early stopping
    es = EarlyStopping( min_delta = EARLY_DELTA, patience = EARLY_PATIENCE )
    if val_idxs is None:
        val_idxs = []

    # train_subset = torch.utils.data.Subset(train_dataset, train_idxs)
    train_weights = [1./train_dataset.class_wts[y] for y in train_dataset[train_idxs][2].argmax(1).tolist() ]
    # logger.debug()
    # logger.debug(train_weights.tolist())
    assert len(train_idxs) == len(train_weights)
    ## number of oversampling
    num_classes = len(train_dataset.class_wts.tolist())
    N_samples = math.ceil( max(train_dataset.class_wts.tolist())/BATCH_SIZE*num_classes)*BATCH_SIZE
    logger.info("wts:{}, #train:{}, #new_train: {}".format(train_dataset.class_wts.tolist(), len(train_idxs), N_samples))

    while epoch < NUM_EPOCHS:
        # logger.debug(("batches", [len(b) for b in batches]))

        if oversample:
            batches = torch.utils.data.DataLoader(
                train_idxs, 
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights = train_weights, 
                    num_samples = N_samples, 
                    replacement=True
                ),
                batch_size = BATCH_SIZE,
            )
        else:
            #if BATCH_SIZE is not None else 1
            batches = np.array_split(np.random.permutation(train_idxs), (len(train_idxs) // BATCH_SIZE) )

        for batch_idxs in batches:
            def closure():
                optimizer.zero_grad()
                train_loss_ = step(model1, train_dataset, batch_idxs, loss, False)[0]
                train_loss_.backward()
                return train_loss_
            optimizer.step(closure)
        try:
            scheduler.step()
        except:
            pass
        train_loss, cm_train, _ = step(model1, train_dataset, train_idxs, loss, True)
        # perf_tr = get_measures(cm_train)
        
        if epoch % VERBOSE_INTERVAL == 0 and VERBOSE:
            logger.info(cm_train)
            logger.info("Train [{:02d}], loss: {:.4g}".format(
                epoch, train_loss.item() ) )
        
        writer.add_scalar('avg_loss/train', train_loss.item(), epoch)
        # writer.add_scalar('prec/train', perf_tr[0][0].item(), epoch)
        # writer.add_scalar('rec/train', perf_tr[1][0].item(), epoch)
        # writer.add_scalar('f1/train', perf_tr[2][0].item(), epoch)
        
        if len(val_idxs) > 0:
            val_loss, cm_val, _ = step(model1, train_dataset, val_idxs, loss, True)
            # perf_val = get_measures(cm_val)
            
            if epoch % VERBOSE_INTERVAL == 0 and VERBOSE:
                logger.info(cm_val)
                logger.info("Val [{:02d}], loss: {:.4g}".format(
                    epoch, val_loss.item() ) )
            writer.add_scalar('avg_loss/val', val_loss.item(), epoch)
            # writer.add_scalar('prec/val', perf_val[0][0].item(), epoch)
            # writer.add_scalar('rec/val', perf_val[1][0].item(), epoch)
            
        writer.flush()
        if len(val_idxs) > 0:
            if es.step( torch.DoubleTensor([val_loss]) ):
                logger.info("early stopping at {} epoch".format(epoch))
                break
            logger.debug( "EARLY_STOPPING epoch:{}, {}".format( epoch, str(es)) )
        epoch += 1
    return model1, epoch

def write_preds(fname, preds, test_dataset):
    preds = preds.tolist()
    try:
        preds = [p[0] for p in preds]
    except:
        pass
    write_header = not os.path.exists(fname)
    with open(fname, "a") as fpreds:
        counter = 1
        targets = test_dataset.df["Target"].values
        tweets = test_dataset.df["Tweet"].values
        # stances = test_dataset.df["Stance"].values
        if write_header:
            fpreds.write("ID\tTarget\tTweet\tStance\n")
        for target, tweet, stance in zip(targets, tweets, preds):
            fpreds.write("{}\t{}\t{}\t{}\n".format( (TARGETs.index(target)+1)*10000+ counter, target, tweet, STANCE_MAP_INV[stance].upper() ))
            fpreds.flush()
            counter += 1

def write_gold(datasets):
    # counter = 10001
    fname = "predictions/goldA.txt"
    with open(fname, "w") as fp:
        fp.write("ID\tTarget\tTweet\tStance\n")
        for name, dataset in datasets.items():
            _, test_dataset = dataset
            counter = 1
            targets = test_dataset.df["Target"].values
            tweets = test_dataset.df["Tweet"].values
            stances = test_dataset.df["Stance"].values
            for target, tweet, stance in zip(targets, tweets, stances):
                fp.write("{}\t{}\t{}\t{}\n".format((TARGETs.index(target)+1)*10000 + counter, target, tweet, stance.upper()))
                counter += 1

def eval_cm(cm):
    counts_preds = cm.sum(axis=0)
    counts_true = cm.sum(axis=1)
    positives = cm.diagonal()
    try:
        recall = np.nan_to_num(positives / counts_true)
        prec = np.nan_to_num(positives / counts_preds)
        f1 = 2 * recall * prec / (recall + prec)
    except:
        logger.error(C)
    return f1

def main():
    datasets = {}
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
    
    for f in glob.glob("predictions/predsA_ub_*.txt"):
        os.remove(f)
    write_gold(datasets)
    
    cms = {}
    fres = open("upper_bound_emb.csv", "w+")
    fres.write("target,model,f_none,f_favor,f_against,official,bacc\n")

    for name, dataset in datasets.items():
        train_dataset, test_dataset = dataset

        train_idxs, val_idxs = train_test_split(
            np.arange(len(train_dataset)), 
            test_size = 0.2, 
            stratify = train_dataset.stance_names,
            random_state = 17
        )
        wts = torch.log(train_dataset.class_wts.sum() / train_dataset.class_wts)
        logger.info("{}: loss-wts:{}".format(name, wts))

        ######### upper_bound model
        for oversample, weighted_loss in [(True, False), (False, True), (False, False)]:
            logger.info("[{}] oversample:{}, weighted_loss:{}".format(name, oversample, weighted_loss))
            loss = torch.nn.CrossEntropyLoss( weight = wts ) if weighted_loss else torch.nn.CrossEntropyLoss( )

            model1 = LinearModel1(input_dims = 768, output_dims = 3).float()
            model1, epochs = train(model1, loss, train_dataset, train_idxs, val_idxs, oversample = oversample, weighted_loss = weighted_loss)
            
            test_loss, cm, preds = step(model1, test_dataset, np.arange(len(test_dataset)), loss, True)
            fname = "predictions/predsA_ub_refit{}_wtL{}_os{}.txt".format(int(False), int(weighted_loss), int(oversample))
            write_preds(fname, preds= preds, test_dataset = test_dataset)
            bacc = (cm.diagonal()/cm.sum(axis=1)).mean()
            stances = [STANCE_MAP_INV.index(s.upper()) for s in test_dataset.df["Stance"].values]

            tmp = "os{}_wtL{}".format(int(oversample), int(weighted_loss))
            cms[tmp] = cms.get( tmp, 0) + cm
            ev = eval_cm(cm)
            msg = "{},{},{:.4g},{:.4g},{:.4g},{:.4g},{:.4g}".format(name, tmp, *ev, np.mean(ev[1:]), bacc  )
            logger.info("METRIC {}".format(msg))
            fres.write(msg + "\n")
            ### refit with entire training data
            logger.info("refitting model w/ %d epochs for %s"%(epochs, name))
            model2 = LinearModel1(input_dims = 768, output_dims = 3).float()
            model2, _ = train(model2, loss, train_dataset, np.arange(len(train_dataset)), NUM_EPOCHS = epochs, oversample = oversample, weighted_loss = weighted_loss)
            test_loss, cm, preds = step(model2, test_dataset, np.arange(len(test_dataset)), loss, True)
            fname = "predictions/predsA_ub_refit{}_wtL{}_os{}.txt".format(int(True), int(weighted_loss), int(oversample))
            write_preds(fname, preds = preds, test_dataset = test_dataset)
            bacc = (cm.diagonal()/cm.sum(axis=1)).mean()

            ev = eval_cm(cm)
            tmp = "os{}_wtL{}_refit".format(int(oversample), int(weighted_loss))
            cms[tmp] = cms.get( tmp, 0) + cm
            
            msg = "{},{},{:.4g},{:.4g},{:.4g},{:.4g},{:.4g}".format(name, tmp, *ev, np.mean(ev[1:]), bacc  )
            logger.info("METRIC {}".format(msg))
            fres.write(msg + "\n")
            # fp.flush()
            fres.flush()
            # logger.info()
    # fp.close()
    for k, cm in cms.items():
        ev = eval_cm(cm)
        bacc = (cm.diagonal()/cm.sum(axis=1)).mean()
        msg = "{},{},{:.4g},{:.4g},{:.4g},{:.4g},{:.4g}".format("all", k, *ev, np.mean(ev[1:]), bacc  )
        logger.info("METRIC {}".format(msg))
        fres.write(msg + "\n")
        # logger.info(eval_cm(v))
        fres.flush()
    fres.close()
if __name__ == "__main__":
    main()