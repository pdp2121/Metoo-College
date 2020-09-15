import os
import sys
import json
import math
from collections import defaultdict

def merge_user_lists(input_dir):
    user_dict = defaultdict(lambda:None)
    for file_name in sorted(os.listdir(input_dir)):
        if file_name.endswith('.txt'):
            college = file_name.split('_followers')[0]
            usernames = open(input_dir+'/'+file_name,'r').readlines()
            usernames = [u.strip() for u in usernames]
            for user in usernames:
                if user_dict[user]:
                    user_dict[user].append(college)
                else:
                    user_dict[user] = [college]
            print('<{}> users added. Total: {}'.format(college, len(user_dict.keys())))

    user_list = []
    for username in user_dict.keys():
        user_info = {"username":username, "college":user_dict[username]}
        user_list.append(user_info)
    return user_list

def write_batches(user_list, output_dir, batch_size):
    user_batch = []
    for i in range(len(user_list)):
        if i % batch_size == 0 and i != 0:
            with open(output_dir+'/batch_{}.json'.format(int(i/batch_size)), 'w') as batch_file:
                print('Writing batch_{}'.format(int(i/batch_size)))
                json.dump(user_batch, batch_file, indent=1)
            user_batch = [user_list[i]]
        else:
            user_batch.append(user_list[i])
            if i == len(user_list)-1:
                with open(output_dir+'/batch_{}.json'.format(math.ceil(i/batch_size)), 'w') as batch_file:
                    print('Writing batch_{}'.format(math.ceil(i/batch_size)))
                    json.dump(user_batch, batch_file, indent=1)

if __name__ == '__main__':
    directory = os.getcwd()
    input_dir = directory+'/college-followers'
    output_dir = directory+'/user-batches'
    BATCH_SIZE = 60000

    user_list = merge_user_lists(input_dir)
    print('*************')
    write_batches(user_list, output_dir, BATCH_SIZE)