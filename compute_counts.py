import json
import os
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime

directory = os.getcwd()
tweet_path = directory + '/metoo-tweets/'
tweet_count = defaultdict(None)
user_count = defaultdict(None)
date_count = defaultdict(None)

metoo_users = []

for file_name in sorted(os.listdir(tweet_path)):
    if file_name.endswith('.json'):
        print('processing '+file_name)
        with open(tweet_path+file_name, 'r') as tweet_batch:
            tweets = json.load(tweet_batch)

            for tweet in tweets:
                for college in tweet['college']:
                    if college not in tweet_count.keys():
                        tweet_count[college] = 1
                    else:
                        tweet_count[college] += 1

                    if college not in user_count.keys():
                        user_count[college] = [tweet['username']]
                    else:
                        if tweet['username'] not in user_count[college]:
                            user_count[college].append(tweet['username'])
                
                date = str(datetime.fromtimestamp(int(tweet['date'])+3600*24)).split(' ')[0]
                if date not in date_count.keys():
                    date_count[date] = 1
                else:
                    date_count[date] += 1

                if tweet['username'] not in metoo_users:
                    metoo_users.append(tweet['username'])

college_data = pd.read_csv(directory+'/college_twitter.csv')
college_data['tweet_count'] = np.nan
college_data['user_count'] = np.nan
for index,row in college_data.iterrows():
    # print(tweet_count[row['twitter']])
    college_data.loc[index,'tweet_count'] = tweet_count[row['twitter']]
    college_data.loc[index,'user_count'] = len(user_count[row['twitter']])

date_data = pd.DataFrame(columns=['date','tweet_count'])
index = 0
for date in sorted(date_count.keys()):
    date_data.loc[index,'date'] = date
    date_data.loc[index,'tweet_count'] = date_count[date]
    index += 1
print(college_data)
print(date_data)
college_data.to_csv(directory+'/college_twitter.csv', index=False)
date_data.to_csv(directory+'/tweet_date.csv', index=False)
print(len(metoo_users))