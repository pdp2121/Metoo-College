import sys
import os
import json
import langdetect
import pandas as pd
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.segmenter import Segmenter
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.dicts.noslang.slangdict import slangdict
import re
import datetime
from collections import defaultdict
import math

def preprocess(text):
    text = text.lower()
    text = re.sub(",", " ", text)
    text = re.sub("@ ", "@", text)
    text = re.sub("# ", "#", text)
    # text = re.sub(r"\'", "", text)
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub(r"\"", "", text)
    text = re.sub("\'", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'\w*pic.twitter.co\w*', '', text)
    text = re.sub(r'\w*twitter.co\w*', '', text)
    text = re.sub(r'\w*twitter.com\w*', '', text)
    text = re.sub(r"./\S+", "", text)
    text = re.sub(r"@ \S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r'\n+', " ", text)
    try:
        if langdetect.detect(text) == 'en':
            return text
        else:
            return ""
    except:
        return ""

def process_tags(text):
    for e in re.findall(r"<hashtag>.*?</hashtag>", text):
        segmented = e.split("<hashtag>")[1].split("</hashtag>")[0].strip()
        hashtag = "_".join(segmented.split(" "))
        text = re.sub(segmented, hashtag, text)
    return text

directory = os.getcwd()
tweet_path = directory + '/metoo-tweets/'
data = pd.DataFrame(columns=['id','conversation','username','text','retweets','favorites','date'])
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'date'],
    # terms that will be annotated
    annotate=['hashtag', 'allcaps', 'elongated', 'repeated', 'emphasis', 'censored'],
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    spell_correction=True,
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons,slangdict]
    )

segmenter = Segmenter(corpus="twitter") 
count = 0
all_texts = []
user_dict = defaultdict(lambda:None)

for file_name in sorted(os.listdir(tweet_path)):
    if file_name.endswith('.json'):
        print('processing '+file_name)
        with open(tweet_path+file_name, 'r') as tweet_batch:
            tweets = json.load(tweet_batch)
            for tweet in tweets:
                # text = preprocess(tweet['content']['text'])
                tokens = text_processor.pre_process_doc(text)
                tokens = [segmenter.segment(t) for t in tokens]
                text = " ".join(tokens)
                text = process_tags(text).strip()
                username = str(tweet['username'])
                if text:
                    if user_dict[username]:
                        user_dict[username] = list(set(user_dict[username])|set(tweet['college']))
                    else:
                        user_dict[username] = tweet['college']

                    data.loc[count,'username'] = str(tweet['username'])
                    data.loc[count,'id'] = str(tweet['id_str'])
                    data.loc[count,'conversation'] = str(tweet['content']['conversation'])
                    data.loc[count,'text'] = text
                    data.loc[count,'retweets'] = int(tweet['interactions']['retweets'])
                    data.loc[count,'favorites'] = int(tweet['interactions']['favorites'])
                    data.loc[count,'date'] = datetime.datetime.fromtimestamp(int(tweet['date'])).strftime('%Y-%m-%d')
                    data.loc[count,'batch'] = file_name
                    count += 1

                    if count%50000 == 0:
                        csv_out = "/metoo_tweets_{}_new.csv".format(math.ceil(count/50000))
                        print("Writing tweets to", csv_out, "...")
                        data.to_csv(directory+csv_out, index=False)
                        data = pd.DataFrame(columns=['id','conversation','username','text','retweets','favorites','date'])
            print(file_name + ' done: ' + str(count))

csv_out = "/metoo_tweets_{}.csv".format(math.ceil(count/50000))
print("Writing tweets to", csv_out, "...")
data.to_csv(directory+csv_out, index=False)

metoo_users = []
for username in user_dict.keys():
    user_info = {"username":username, "college":user_dict[username]}
    metoo_users.append(user_info)
print('metoo_users:', len(metoo_users))

with open(directory+'/metoo_users.json', 'w') as metoo_file:
    json.dump(metoo_users, metoo_file, indent=2)

# tweet_by_date = data.groupby('date').count()
# tweet_by_date.to_csv(directory+"/date.csv", index=False)