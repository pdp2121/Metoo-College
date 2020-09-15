import pandas as pd

directory='/home/vietduong/Desktop/metoo_project/'
df_metoo_topics = pd.read_csv(directory+'metoo_topics.csv')

for topic in range(50):
    df = df_metoo_topics.loc[df_metoo_topics['Dominant_Topic']==topic]
    open(directory+'clusters/cluster_{}.txt'.format(topic), 'w').close()
    for tweet in df['Tweet']:
        with open(directory+'clusters/cluster_{}.txt'.format(topic), 'a') as f:
            f.write(str(tweet).strip()+'\n')