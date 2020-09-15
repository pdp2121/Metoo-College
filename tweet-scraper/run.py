import sys
import csv
import time
import os
import json
import math
import got
from got.Tweet import tweet_to_dict
import progressbar

def get_tweets(input_dir, output_dir, max_tweets, search_phrase, since, until):
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.json'):
            print("Getting tweets from <{}>.".format(file_name))
            users = json.load(open(input_dir+'/'+file_name, 'r'))

            with progressbar.ProgressBar(max_value=len(users)) as bar:
                for i in range(len(users)):
                    config = got.TweetCriteria()
                    config.setUsername(users[i]["username"])
                    config.setSince(since)
                    config.setUntil(until)
                    if search_phrase:
                        config.setQuerySearch(search_phrase)
                    if max_tweets:
                        config.setMaxTweets(max_tweets)
                    try:
                        tweets = got.TweetManager.getTweets(config)
                    except Exception as e:
                        print(e)
                        pass
                    if tweets:
                        tweets = [tweet_to_dict(t) for t in tweets]
                        for tweet in tweets:
                            tweet["college"] = users[i]["college"]
                        tweet_batch = []
                        try:
                            tweet_batch = json.load(open(output_dir+'/'+file_name, 'r'))
                        except:
                            open(output_dir+'/'+file_name, 'w').close()
                        tweet_batch.extend(tweets)
                        with open(output_dir+'/'+file_name, 'w') as tweet_file:
                            json.dump(tweet_batch, tweet_file, indent=2)
                    bar.update(i)

            print("{} done".format(file_name))
            # print("Writing {}> done.".format(file_name))
            print("Sleeping to avoid rate limit.")
            for i in progressbar.progressbar(range(60*15)):
                time.sleep(1)

if __name__ == '__main__':
    directory = os.getcwd()
    input_dir = directory + '/user-batches'
    output_dir = directory + '/metoo-tweets'
    get_tweets(input_dir, output_dir, None, "metoo", "2017-10-15", "2017-11-15")