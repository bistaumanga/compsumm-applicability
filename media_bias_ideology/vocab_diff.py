import json, sys, time,os
import pandas as pd
import numpy as np
from collections import Counter
from commons.utils import get_logger
from timeit import default_timer as timer
import nltk

from commons.functions import js_div_vecs, multinomial_llr_vecs
from sklearn.feature_extraction.text import CountVectorizer as BoW
from sklearn.feature_extraction.text import TfidfVectorizer

from itertools import product, chain
from commons.tokenizer import word_tokenize1

PROJ_ROOT = os.environ.get("PROJ_ROOT","./")
# field = sys.argv[1]
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
N_SENTS = 3 #or "None" for all sents
if type(N_SENTS) is int:
    assert N_SENTS >= 1
else:
    N_SENTS = None

logger = get_logger("Vocab-Diff")

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

def tokenize(row):
    title_toks = word_tokenize1(row["title"])
    des_toks = list(chain(*[word_tokenize1(s) for s in row["sents"] ] ))
    return title_toks + des_toks

def read_data(topic, extract_tokens = True):
    df = pd.read_json("{}/dataset/ents_filtered_{}_sm.json".format(PROJ_ROOT, topic), orient = "records", lines = True)
    df.drop([ "domain", "url" ], inplace = True, axis = 1)

    logger.info("data read, #rows:{}".format(len(df)))
    logger.info("Distribution: {}".format(Counter(df["ideology"])))
    
    df = df.query("ideology in @LABELS").reset_index()
    df["sents"] = [ s[:N_SENTS] for s in df["sents"] ]
    # print(df["nouns"].values)
    if extract_tokens:
        df["tokens"] = df.apply(tokenize, axis = 1)
        df["nouns"] = df["nouns"].apply(lambda nouns: [str(n).lower() for n in nouns])
        df["n_toks"] = df.apply(lambda row: len(row["tokens"]), axis = 1 )
        # df.drop([ "title", "sents" ], inplace = True, axis = 1)
    return df

def run(df1, label1, label2, field):
    idx1 = np.where(df1.ideology == label1 )[0]
    idx2 = np.where(df1.ideology == label2 )[0]
    bow = TfidfVectorizer( tokenizer = lambda x: x, 
                    lowercase = False, min_df = 2, 
                    preprocessor = None, use_idf = False, 
                    norm = None,
                    binary = False,
                    max_df = 0.2 if field == "tokens" else 0.5)
    X = bow.fit_transform( df1.iloc[idx1.tolist() + idx2.tolist()][field] )
    
    x1 = np.array(X[:len(idx1), :].sum(axis = 0)).squeeze()
    x2 = np.array(X[len(idx1):, :].sum(axis = 0)).squeeze()
    cos = x1.dot(x2) / (np.sqrt(x1.dot(x1)) * np.sqrt(x2.dot(x2)))
    # print(field, cos)

    # c1 = Counter(list(chain(*df1.iloc[idx1][field].values)))
    # c2 = Counter(list(chain(*df1.iloc[idx2][field].values)))
    # c = c1 + c2
    # js = js_div(c1, c2)
    # vocab = c.keys()
    js = js_div_vecs(x1, x2)
    llr = multinomial_llr_vecs(x1, x2)

    vocab = np.array(bow.get_feature_names())
    assert len(vocab) == len(x1) and len(vocab) == len(x2)
    # if field == "nouns":
    #     print(vocab)
    #     assert 1 == 2
    top1 = vocab[np.argpartition(x1, -10)[-10:]]
    top2 = vocab[np.argpartition(x2, -10)[-10:]]
    return len(x1), js, len(idx1), len(idx2), llr, cos, top1, top2

def main():
    fp = open("vocab_diff.csv", "w")
    fp.write("topic,month,comp,n1,n2,units,vocab,jsd,cos,llr,top1,top2\n")
    logger.info("comps:{}".format(COMPS))
    for topic in ("guncontrol", "cbp01", "cbp02", "climatechange", "chr01"):
    
        df = read_data(topic, True )
        logger.info("topic %s"%topic)
        
        for (label1, label2) in COMPS:
            #
            vocab_t, js_t, n1t, n2t, llr_t, cos_t, t1, t2 = run(df, label1, label2, "tokens")
            vocab_n, js_n, n1n, n2n, llr_n, cos_n, n1, n2 = run(df, label1, label2, "nouns")
            assert n1t == n1n and n2t == n2n
            msg = "{},{},{}_{},{},{},{},{},{:.6g},{:.6g},{:.6g},{},{}".format(
                TOPIC_MAP[topic], "all",
                label1, label2, n1t, n2t, 
                "tokens", vocab_t, js_t, cos_t, llr_t,
                ";".join(t1).replace(",", ""), ";".join(t2).replace(",", "")
            )
            logger.info(msg)
            fp.write(msg + "\n")

            msg = "{},{},{}_{},{},{},{},{},{:.6g},{:.46},{:.6g},{},{}".format(
                TOPIC_MAP[topic], "all",
                label1, label2, n1n, n2n, 
                "nouns", vocab_n, js_n, cos_n, llr_n,
                ";".join(n1).replace(",", ""), ";".join(n2).replace(",", "")
            )
            logger.info(msg)
            fp.write(msg + "\n")
            

        for dr in dates:
            from_, to_ = dr
            df1 = df.query('date >= @from_ and date <= @to_').reset_index(drop=True)
                
            logger.info( "topic:{}, {}:{}, #articles:{}, {}".format(topic, from_, to_, len(df1), Counter(df1["ideology"]) ) )

            for (label1, label2) in COMPS:
                
                vocab_t, js_t, n1t, n2t, llr_t, cos_t, t1, t2 = run(df1, label1, label2, "tokens")
                vocab_n, js_n, n1n, n2n, llr_n, cos_n, n1, n2 = run(df1, label1, label2, "nouns")
                assert n1t == n1n and n2t == n2n
                msg = "{},{},{}_{},{},{},{},{},{:.6g},{:.6g},{:.6g},{},{}".format(
                    TOPIC_MAP[topic], MONTHS_MAP[from_[:7]],
                    label1, label2, n1t, n2t, 
                    "tokens", vocab_t, js_t, cos_t, llr_t,
                    ";".join(t1).replace(",", ""), ";".join(t2).replace(",", "")
                )
                logger.info(msg)
                fp.write(msg + "\n")

                msg = "{},{},{}_{},{},{},{},{},{:.6g},{:.6g},{:.6g},{},{}".format(
                    TOPIC_MAP[topic], MONTHS_MAP[from_[:7]],
                    label1, label2, n1n, n2n, 
                    "nouns", vocab_n, js_n, cos_n, llr_n,
                    ";".join(n1).replace(",", ""), ";".join(n2).replace(",", "")
                )
                logger.info(msg)
                fp.write(msg + "\n")
    fp.close()
    logger.info("Done")

if __name__ == "__main__":
    main()