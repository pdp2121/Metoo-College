import os
import sys
import json
import time
import math
from tweepy import Cursor
from twitter_client import get_twitter_client
import progressbar

# def usage():
#     print("Usage:")
#     print("python {} <username>".format(sys.argv[0]))

def paginate(items, n):
    # enerate n-sized chunks from items
    for i in range(0, len(items), n):
        yield items[i:i+n]

def get_followers(username, max_followers, output_file):
    print("Getting <{}> followers".format(username))
    client = get_twitter_client()
    max_pages = math.ceil(max_followers/5000)
    count = 0
    json_file = output_file.split('.')[0] + '_full.json'
    with open(json_file, 'w') as json_output:
        with progressbar.ProgressBar(max_value=progressbar.UnknownLength) as bar:
            for followers in Cursor(client.followers_ids, screen_name=username).pages(max_pages):
                for chunk in paginate(followers, 100):
                    try:
                        users = client.lookup_users(user_ids=chunk)
                        for user in users:
                            user_info = user._json
                            screen_name = user_info['screen_name']
                            with open(output_file, 'a') as txt_output:
                                txt_output.write(screen_name+'\n')
                            json_output.write(json.dumps(user._json)+'\n')
                            count += 1
                            bar.update(count)
                    except:
                        pass
                if len(followers) == 5000:
                    time.sleep(60)
    print("<{}> followers completed".format(username))
    time.sleep(60)

if __name__ == '__main__':
    directory = os.getcwd()
    usernames = open(directory+'/usernames.txt','r').readlines()
    usernames = [u.strip() for u in usernames]

    MAX_FOLLOWERS = 1500000

    for username in usernames:
        file_name = directory + '/college-followers/{}_followers.txt'.format(username)
        get_followers(username, MAX_FOLLOWERS, file_name)