import re, os
import nltk
import numpy as np
# from segtok.segmenter import split_multi
# from segtok.tokenize import web_tokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
# from commons.tokenizer
from pyhocon import ConfigFactory
from functools import reduce
import nltk.data

## define sentence and word tokenizers here
# word_tokenizer = web_tokenizer
# sent_tokenizer = split_multi
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# sent_tokenizer = lambda text: tokenizer.tokenize(text, realign_boundaries=True)

conf = ConfigFactory.parse_file(os.path.dirname(os.path.realpath(__file__)) + "/text.conf")

from commons.utils import get_logger
logger = get_logger("cleaner")

SENTENCE_THRESH_LB = conf.get("article_filter.threshold.sentences.words.lb")
SENTENCE_THRESH_UB = conf.get("article_filter.threshold.sentences.words.ub")
NUM_SENTS_THRESH = conf.get("article_filter.threshold.description.sentences")
DESCRIPTION_THRESH = conf.get("article_filter.threshold.description.chars")
NUM_WORDS_THRESH = conf.get("article_filter.threshold.description.words")
DEDUP_SIM_THRESH_SENT_SENT = conf.get("article_filter.threshold.dedup.sent_sent")
DEDUP_SIM_THRESH_TITLE_SENT = conf.get("article_filter.threshold.dedup.title_sent")
DEDUP_SIM_THRESH_DOC_DOC = conf.get("article_filter.threshold.dedup.doc_doc")
DEDUP_SIM_WORDS_DIFF = conf.get("article_filter.threshold.dedup.description.words.diff")

TITLE_WORDS_THRESH = conf.get("article_filter.threshold.title.words")

JUNK_SENT = [txt.lower().strip() for txt in conf.get("article_filter.JUNK_SENT")]
JUNK_SENT_REGEX = [re.compile(txt.strip()) for txt in conf.get("article_filter.JUNK_SENT_REGEX")]
JUNK_TEXT = [txt.lower().strip() for txt in conf.get("article_filter.JUNK_TEXT")]
JUNK_TITLE = [txt.lower().strip() for txt in conf.get("article_filter.JUNK_TITLE")]
REPLACES_SENT_REGEX = [re.compile(txt.strip()) for txt in conf.get("article_filter.REPLACES_SENT_REGEX")]

## process for quotes
def process_text(text, process_quotes = True, acronyms = True):
    ## non standard characters
    text = text.replace("''", '"').replace("``", '"').replace('_', ' ').replace('–', '-')
    text = re.sub('(\'\'|\“|\”)', '"', text).replace("’", "'").replace('`', '\'')
    # [^\+\%;\-\=\_:$A-Za-z\d\s\n\,``\"\'\(\.\)\\\/\?#\*!\&@]
    text = text.replace('…', '...').replace("\\'", "'").replace(".\"-", "\"-").replace('\t', ' ')
    text = text.replace('``', '"').replace('\'\'', '"').replace('\n', ' ').strip("'<>() ")
    text = re.sub(r'\"([0-9a-z\-A-Z]+)(\.?)\"', r'\1\2 ', text) ## single words inside double quotes
    text = re.sub(r'\'([0-9a-z\-A-Z]+)(\.?)\'', r'\1\2 ', text) ## single words inside single quotes
    # text= re.sub(r'([a-z])\-([a-z])\s', r'\1 \2 ', text)

    if acronyms:
        text = re.sub(r'([ap])\.m\.\s', r"\1m ", text) ## a.m. and p.m.
        text = re.sub(r'([A-Z]\.){2,}\s', lambda match: re.sub(r"\.", "", match.group()) + ' ', text) ## acronyms
        text = re.sub(r'U[;\.]\s*([SK])[;\.]', r'U\1', text)
        text = re.sub(r'([a-zA-Z])\.([^\d\.\sa-z\,])', r'\1. \2', text) ## insert . for sentence tokenization
    
    if process_quotes:
        text = re.sub(r'\"([^\"]+)\"', lambda match: re.sub(r"\.+\s*", "; ", match.group()), text) ## . inside quotes
        # text = re.sub(r'\'([^\']+)\'', lambda match: re.sub("\.+\s*", "; ", match.group()), text)
        text = re.sub(r';+\s+\"', '."', text)
        text = re.sub(r'Mr;', 'Mr ', text)
    text = re.sub(r'\s{2, }', ' ', text)
    return text

def process_sent(text):
    # text = re.sub(r'(?<!\w)([A-Za-z])\.', r'\1', text.strip())
    text = re.sub(r'\"([^\"]+)\"', lambda match: re.sub(r"\.+\s*", "; ", match.group()), text.strip()) ## . inside quotes
    text = re.sub(r';+\s+\"', '."', text)
    text = re.sub(r'\\\'', '\'', text)
    return text

## similarity metric to detect duplicates
def jaccard_sim(text1, text2):
    if type(text1) is str:
        text1 = set(word_tokenize(text1.lower()))
    if type(text2) is str:
        text2 = set(word_tokenize(text2.lower()))
    return len(text1 & text2) / len(text1 | text2), len(text1) > len(text2)

def valid_sentence(s):
    words = [w for w in word_tokenize(s) if w.isalpha()]
    num_words = len(words)
    if num_words < SENTENCE_THRESH_LB or num_words > SENTENCE_THRESH_UB:
        return False, num_words
    for junk_sent in JUNK_SENT:
        if junk_sent in s.lower():
            logger.info("<< spam-str: " + junk_sent + " >> " + s)
            return False, num_words
    for rex in JUNK_SENT_REGEX:
        if re.search(rex, s) is not None:
            logger.info("<< spam-rex: " + rex.pattern + " >> " + s)
            return False, num_words
    return True, num_words

def words_list(sentences):
    return reduce( lambda a,b: a + b, 
        map ( lambda sent: word_tokenize(sent.lower()) , sentences ))

def sim_sentences_list(sentences1, sentences2):
    words1 = words_list(sentences1)
    words2 = words_list(sentences2)
    sim_text, _ = jaccard_sim(set(words1), set(words2))
    return sim_text, abs(len(words1) - len(words2)) 

def jaccard_dist_F1_news(idx1, idx2, **kwargs):
    ix1 = int(idx1[0])
    ix2 = int(idx2[0])
    title_sim, _ = jaccard_sim(kwargs["dataset"].titles[ix1], kwargs["dataset"].titles[ix2])
    text_sim, _ = sim_sentences_list(kwargs["dataset"].sents[ix1], kwargs["dataset"].sents[ix2])
    print((idx1[0], idx2[0]), end = ";")
    return 1. -  2 * title_sim * text_sim / (title_sim + text_sim)

def print_file(fp, prefix, title, description = None, sents = None):
    assert description is not None or sents is not None
    text = description or "\n".join([s[0] for s in sents])
    fp.write(prefix + " << " + title + \
            " >>\n\n" + text + "\n\n" + "#" * 50 + "\n\n" )

def filter_article(title, description, fdups = None, ffiltered = None, dedup_title = False):

    ## check title spam filtering, should have done in ES
    for jt in JUNK_TITLE:
        if jt.lower() in title.lower():
            if ffiltered: print_file(ffiltered, "title-junk", title, description = description)
            return None, None
    ## check description spam filtering, should have done in ES
    for jt in JUNK_TEXT:
        if jt.lower() in description.lower():
            if ffiltered: print_file(ffiltered, "text-junk", title, description = description)
            return None, None
    
    ## check if title is too short
    title = title.replace("\"", "'").replace("“", "'").replace("’", "'").replace("–", "-").replace("”", "'").replace("‘", "'")
    if len(word_tokenize(re.sub(r'[^\w\s]+', '', title) )) < TITLE_WORDS_THRESH :
        if ffiltered: print_file(ffiltered, "#title", title, description = description) 
        return None, None  

    des = process_text(description)
    
    ## check description threshold
    if len(des) < DESCRIPTION_THRESH:
        if ffiltered: print_file(ffiltered, "len(text)", title, description = description)
        return title, None

    sents = []
    for s in sent_tokenize(des):
        s = process_sent(s)
        valid, num_words = valid_sentence(s.lower()) ## check if sentence is good enough
        if valid: 
            ## some filtering to remove unnecessary phrases
            for rep in REPLACES_SENT_REGEX:
                if re.search(rep, s) is not None: 
                    logger.debug("<< replace-rex: " + rep.pattern + " >> " + s)
                    s = re.sub(rep, ' ', s)
            s = re.sub(r'(\s[A-Za-z\d]+)-([A-Za-z]+)\s', r' \1 \2 ', s)## word1-word2 becomes word1 word2
            s = re.sub(r'\s+', ' ', s) ## remove unnecessary whitespaces
            ## create space between numbers and alphabets
            s = re.sub(r'([A-Za-z]+)(\d)', r'\1 \2', s)
            s = re.sub(r'(\d)([A-Za-z]{3,})', r'\1 \2', s)

            sents.append((s, num_words))
        
    ## too few sentences
    if len(sents) < NUM_SENTS_THRESH:
        if ffiltered: print_file(ffiltered, "#sents", title, sents = sents)
        return title, None

    ## is title similar to first sentence
    if dedup_title:
        score, _ = jaccard_sim(title, sents[0][0])
        if (title in sents[0][0] and score >= DEDUP_SIM_THRESH_TITLE_SENT):
            if fdups is not None: fdups.write("title:-{:.3f}\n\t-{} \n\t-{}\n".format(score, title, sents[0][0]))
            del sents[0]

    dups = []

    ## deduplicate adjacent sentences
    for i in range(len(sents) -1):
        score, first_long = jaccard_sim(sents[i][0], sents[i+1][0])
        if score >= DEDUP_SIM_THRESH_SENT_SENT:
            dupidx = i+1 if first_long else i
            dups.append(dupidx)
            if fdups is not None: 
                fdups.write("sentences:-{:.3f} << {} >>\n\t-{} \n\t-{}\n".format(
                    score, title, sents[i], sents[i+1]))
    sents = [s for (i, s) in enumerate(sents) if i not in dups]

    ## too few words
    if sum([s[1] for s in sents]) < NUM_WORDS_THRESH:
        if ffiltered: print_file(ffiltered, "|text|", title, sents = sents)
        return title, None
    
    ## too few sentences
    if len(sents) < NUM_SENTS_THRESH:
        if ffiltered: print_file(ffiltered, "#sents", title, sents = sents)
        return title, None

    return title, [s[0] for s in sents]