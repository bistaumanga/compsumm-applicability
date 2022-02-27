import torch
import numpy as np
from collections import Counter
from torch.utils.data.dataset import Dataset
import pandas as pd

from transformers import AutoModel, AutoTokenizer 
from commons.utils import get_logger
import pickle, sys, time, warnings

warnings.filterwarnings('ignore')
logger = get_logger("Data")
models = None

STANCE_MAP = {
    "NONE": [1, 0, 0],
    "FAVOR": [0, 1, 0],
    "AGAINST": [0, 0, 1]
}
STANCE_MAP_INV = ["NONE", "FAVOR", "AGAINST"]

class TweetStanceDataset(Dataset):
    def __init__(self, path, target = None):
        df = pd.read_csv(path, encoding= 'unicode_escape', engine='python')
        if target == "All":
            target = None
        logger.info("read data w/ {} rows and columns:{}".format(
            len(df), df.columns
        ))
        self.__filtered = "All"
        if target != None:
            self.__filtered = target
            df = df.query("Target == @target")
        
        self.__df = df
        # self.__tweets = df["Tweet"].values
        self.__stances = torch.LongTensor( [STANCE_MAP[s] for s in df["Stance"] ])
        
        # self.__targets = df["Target"]
        assert len(self.__df) == len(self.__stances)
        self.__featurize()
        logger.info(self)

    def __len__(self):
        return len(self.__tweets)
    
    @property
    def df(self):
        return self.__df

    @property
    def target(self):
        return self.__filtered
    
    @property
    def stance_names(self):
        return self.__df["Stance"].values

    @property
    def class_wts(self):
        y = self.__stances.argmax(1).tolist()
        cc = Counter(y)
        # print(cc)
        return torch.FloatTensor([cc[0], cc[1], cc[2]])

    def __repr__(self):
        return pd.crosstab(self.__df["Target"], self.__df["Stance"], margins = self.__filtered == "All").to_string()

    @property
    def emb(self):
        return self.__pooled_feats

    ## don't want to wait 5 minutes in every epoch
    def __featurize(self):
        ''' 
            caching the BERTweet features
        '''
        global models
        try:
            bertweet, tokenizer = models
        except:
            start_model_read = time.time()
            bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
            tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
            models = bertweet, tokenizer
            logger.info("BERTweet loaded in {:.1f} secs".format(time.time() - start_model_read))
        
        start = time.time()
        feats, pooled_feats = list(), list()
        i = 0
        for tweet in self.__tweets:
            i += 1
            input_ids = torch.tensor([tokenizer.encode( tweet ) ])
            with torch.no_grad():
                features = bertweet(input_ids)  # Models outputs are now tuples
            pooled_feats.append(features[1][0] )
            feats.append(features[0][0])
            if (i % 500) == 0:
                logger.info("featurized %d tweets" %i)

        self.__pooled_feats = torch.stack(pooled_feats)
        logger.info("pooled feats: {}".format(self.__pooled_feats.shape))
        self.__feats = feats
        logger.info("featurized {} tweets in {:.1f} secs".format(i, time.time() - start ))

    def __getitem__(self, idx):
        if type(idx) == int or type(idx) == np.int64:
            idx = [idx]
        # return a 5 tuple
        return ( [self.__tweets[ix] for ix in idx ], # tweet
                    self.__targets.values[idx], # target
                    self.__stances[idx], # stance label, one hot
                    self.__pooled_feats[idx], # pooled features rep
                    [self.__feats[ix] for ix in idx] ## features rep for each token
                )
    
    def save(self, root, dataset):
        fname = "{}/{}_{}.pkl".format(root, dataset, self.__filtered).replace(" ", "_")
        with open(fname, "wb") as fp:
            pickle.dump(self, fp)
            return
    
    @staticmethod
    def load(root, dataset, target = "All"):
        fname = "{}/{}_{}.pkl".format(root, dataset, target).replace(" ", "_")
        with open(fname, "rb") as fp:
            return pickle.load(fp)

def main():
    dataset = sys.argv[1]
    TARGETs = ["Atheism",
        "Climate Change is a Real Concern",
        "Feminist Movement",
        "Hillary Clinton",
        "Legalization of Abortion"
    ]
    TARGETs = ["All"]

    for target in TARGETs:
        try:
            ds = TweetStanceDataset.load("cache", dataset, target)
            logger.info(ds)
        except:
            logger.info("caching %s ..."%target)
            ds = TweetStanceDataset("./data/StanceDataset/%s.csv"%dataset, target = target)
            print(ds)
            ds.save("cache", dataset)
            logger.info("done caching %s !"%target)

if __name__ == "__main__":
    main()