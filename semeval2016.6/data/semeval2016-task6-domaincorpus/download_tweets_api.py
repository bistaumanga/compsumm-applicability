import sys
import os
import time
import datetime
import argparse

from twitter import *

parser = argparse.ArgumentParser(description="downloads tweets")
parser.add_argument('--partial', dest='partial', default=None, type=argparse.FileType('r'))
parser.add_argument('--dist', dest='dist', default=None, type=argparse.FileType('r'), required=True)
parser.add_argument('--output', dest='output', default=None, type=argparse.FileType('w'), required=True)
args = parser.parse_args()


CONSUMER_KEY='oGPBqxuPnbtKMNcCaz3Gn01Ag'
CONSUMER_SECRET='SsXZJgdubX9PQlBNRRaRPwopgu6ClL1j9TAMPImULGb9vGtB9y'
OAUTH_TOKEN = "28290853-on37QimemYcKeCH0ZkzhseoQzgOs4c1n5yH7dcFOm"
OAUTH_SECRET = "h8UIDIGx7XOBWl6Lrzl2seR9zTAEGGT9z8q3MI47KDjXT"

# MY_TWITTER_CREDS = os.path.expanduser('~/.my_app_credentials')
# if not os.path.exists(MY_TWITTER_CREDS):
#     oauth_dance("Semeval sentiment analysis", CONSUMER_KEY, CONSUMER_SECRET, MY_TWITTER_CREDS)
# oauth_token, oauth_secret = read_token_file(MY_TWITTER_CREDS)
t = Twitter(auth=OAuth(OAUTH_TOKEN, OAUTH_SECRET, CONSUMER_KEY, CONSUMER_SECRET))

cache = {}
if args.partial != None:
    for line in args.partial:
        fields = line.strip().split("\t")
        text = fields[-1]
        sid = fields[0]
        cache[sid] = text

for line in args.dist:
    fields = line.strip().split('\t')
    sid = fields[0]

    while not sid in cache:
        try:
            text = t.statuses.show(_id=sid)['text'].replace('\n', ' ').replace('\r', ' ')
            cache[sid] = text.encode('utf-8')
        except TwitterError as e:
            if e.e.code == 429:
                rate = t.application.rate_limit_status()
                reset = rate['resources']['statuses']['/statuses/show/:id']['reset']
                now = datetime.datetime.today()
                future = datetime.datetime.fromtimestamp(reset)
                seconds = (future-now).seconds+1
                if seconds < 10000:
                    sys.stderr.write("Rate limit exceeded, sleeping for %s seconds until %s\n" % (seconds, future))
                    time.sleep(seconds)
            else:
                msg = "ERROR: " + e.response_data['errors'][0]['message']
                cache[sid] = msg
        except Exception as e:
            msg = "ERROR: " + str(e)
            cache[sid] = msg
    try:
        text = cache[sid].decode("utf-8")
    except:
        text = cache[sid]
    # print(cache)
    args.output.write("\t".join(fields + [text]) + '\n')
    args.output.flush()
