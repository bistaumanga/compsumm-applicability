from collections import Counter 
import numpy as np 
from sklearn.metrics import confusion_matrix

def bacc(y_true, y_pred, verbose = False):
    C = confusion_matrix(y_true, y_pred)
    if verbose:
        print("bacc", Counter(y_true), Counter(y_pred))
        print(C)
    counts = C.sum(axis=1)
    positives = C.diagonal()
    res = np.mean(positives / counts)
    return res

def macro_f1(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    counts_preds = C.sum(axis=0)
    counts_true = C.sum(axis=1)
    positives = C.diagonal()
    recall = np.mean(positives / counts_true)
    prec = np.mean(positives / counts_preds)
    return 2.0 * recall * prec / (recall + prec)

def each_f1(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    counts_preds = C.sum(axis=0)
    counts_true = C.sum(axis=1)
    positives = C.diagonal()
    try:
        recall = positives / counts_true
        prec = positives / counts_preds
        f1 = 2 * recall * prec / (recall + prec)
    except:
        print(C)
    return f1

def avg_f1(y_true, y_pred):
    return np.mean(each_f1(y_true, y_pred))

def multinomial_llr_vecs(k1, k2):
    p1 = k1 / sum(k1)
    p2 = k2 / sum(k2)
    q = (k1 + k2) / sum(k1 + k2) ## this is diff w/ jsd
    return 2 * (
        np.sum(k1 * np.log(p1/q, out=np.zeros_like(p1), where=(p1!=0))) + 
        np.sum(k2 * np.log(p2/q, out=np.zeros_like(p2), where=(p2!=0)))
    )

def js_div_vecs(k1, k2):
    p1 = k1 / sum(k1)
    p2 = k2 / sum(k2)
    q = (p1 + p2) / 2.0 ## this is diff w/ llr
    kl_p1q = np.sum(p1 * np.log(p1/q, out=np.zeros_like(p1), where=(p1!=0)))
    kl_p2q = np.sum(p2 * np.log(p2/q, out=np.zeros_like(p2), where=(p2!=0)))
    return 0.5 * (kl_p1q + kl_p2q)

def js_counters(cA, cB):
    keys = sorted(list((cA | cB).keys()))
    k1 = np.array([ cA[k] for k in keys ])
    k2 = np.array([ cB[k] for k in keys ])
    return js_div_vecs(k1, k2)

def kl_counters(cA, cB):
    if sizeA is None:
        sizeA = sum(cA.values())
    if sizeB is None:
        sizeB = sum(cB.values())
    
    ## kl div is not defined if q_w = 0 for some w, so we ignore those
    keys = (cA & cB).keys()
    res = 0.0
    for key in keys:
        if cA[key] != 0:
            res += ( cA[key] / sizeA * np.log2(cA[key] / sizeA * sizeB / cB[key]) )
    return res


def softmax(x):
	z = np.exp( x - np.max(x) )
	return z / sum(z)