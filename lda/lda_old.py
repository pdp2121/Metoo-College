import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
import math

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatize
import spacy

# Plotting tools
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
extra_stop_words = ['from', 'subject', 're', 'edu', 'use','me_too', 'metoo','not','do',"arent", "couldnt", "dont", "didnt", "doesnt", 
                "hadnt", "hasnt", "havent", "isnt", "mightnt","mustnt", "neednt", "shant","shouldnt", "wasnt", "werent", 
                "wont", "wouldnt",'cannot','cant']
stop_words.extend(extra_stop_words)

import pyLDAvis.gensim

def tokenize(sentence):
        return gensim.utils.simple_preprocess(str(sentence).lower(), deacc=True)  # deacc=True removes punctuations

# functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatize(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# prepare data for lda
directory = os.getcwd()
for i in range(1,8,1):
    data = pd.read_csv(directory+'/metoo_tweets_{}.csv'.format(i))
    if i==1:
        texts = list(data['text'])
    else:
        texts.extend(list(data['text']))

print("Read {} tweets".format(len(texts)))
texts = [re.sub(r"<.*?>", "", t) for t in texts]
texts = [re.sub("metoo", "", t) for t in texts]
texts = [t.strip() for t in texts]
tokens = [tokenize(t) for t in texts]

# remove stop words
tokens_nostops = remove_stopwords(tokens)
print(tokens_nostops[:1])

# build the bigram and trigram models
# higher threshold fewer phrases.
bigram = gensim.models.Phrases(tokens, min_count=10, threshold=100) 
trigram = gensim.models.Phrases(bigram[tokens], threshold=100)

# faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# print trigram example
print(trigram_mod[bigram_mod[tokens[0]]])

# Form Bigrams
tokens_bigrams = make_bigrams(tokens_nostops)

# initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatize keeping only noun, adj, vb, adv
lemmatized = lemmatize(tokens_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
lemmatized_copy = lemmatized
lemmatized = list(filter(None, lemmatized))

# create dictionary
id2word = corpora.Dictionary(lemmatized)
print(len(id2word))
# create corpus
# term document frequency
corpus = [id2word.doc2bow(t) for t in lemmatized]
print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
# tf-idf transformation
from gensim.models import TfidfModel
tfidf = TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

# set mallet path
# mallet_path = directory+'/mallet-2.0.8/bin/mallet'

def train_models(dictionary, corpus, texts, limit, start, step):
        coherence_values = []
        num_topics = []
        models = []
        for n in range(start, limit, step):
                model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                        id2word=dictionary,
                                                        num_topics=n,
                                                        random_state=100,
                                                        iterations=100,
                                                        update_every=1,
                                                        chunksize=5000,
                                                        passes=10,
                                                        alpha='auto',
                                                        per_word_topics=False)
                              
                models.append(model)
                num_topics.append(n)
                coherence_model = CoherenceModel(model=model, texts=lemmatized, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherence_model.get_coherence())
                print(str(n)+', coherence = '+str(coherence_model.get_coherence())+', perplexity = '+str(model.log_perplexity(corpus)))

                # visualize the topics
                vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
                pyLDAvis.save_html(vis,directory+'/vis/lda_{}.html'.format(n))

        return models, num_topics, coherence_values

# experiment
limit = 50
start = 20
step = 5
models, num_topics, coherence_values = train_models(dictionary=id2word, corpus=corpus_tfidf, texts=lemmatized, start=start, limit=limit, step=step)

# show graph
plt.plot(num_topics, coherence_values)
plt.xlabel("num_topics")
plt.ylabel("coherence")
plt.show()

# # build Mallet model
# lda_mallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus_tfidf, num_topics=50, id2word=id2word)

# # show Topics
# pprint(lda_mallet.show_topics(formatted=False))

# # visualize the topics
# lda_model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(lda_mallet)
# vis = pyLDAvis.gensim.prepare(lda_model, corpus_tfidf, id2word)
# pyLDAvis.save_html(vis,'LDA.html')

# # compute Coherence Score
# coherence_model_lda_mallet = CoherenceModel(model=lda_mallet, texts=lemmatized, dictionary=id2word, coherence='c_v')
# coherence_lda_mallet = coherence_model_lda_mallet.get_coherence()
# print('\nCoherence Score: ', coherence_lda_mallet)

def format_topics_sentences(ldamodel, corpus, texts):
    # initial output
    sent_topics_df = pd.DataFrame()

    # get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # print(row)
        # get the dominant topic, perc contribution and keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                        break

    sent_topics_df.columns = ['dominant_Topic', 'perc_contribution', 'topic_keywords']

    # add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


df_topic_sents_keywords = format_topics_sentences(ldamodel=models[0], corpus=corpus_tfidf, texts=lemmatized)

# format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['id', 'dominant_topic', 'perc_contribution', 'topic_keywords', 'text']

# save
data_copy = [sent.replace(',','') for sent in data]
lemmatized = [''.join(map(str, lemmatized_copy)) for l in lemmatized_copy]
raw_lemmatized = pd.DataFrame(columns=['text','tweet'])
raw_lemmatized['text'] = lemmatized_copy
raw_lemmatized['tweet'] = data_copy
print('processing output')
df_dominant_topic = pd.merge(df_dominant_topic, raw_lemmatized, on='text')
df_dominant_topic.to_csv(directory+'metoo_topics.csv')


for topic in range(20):
    df = df_dominant_topic.loc[df_dominant_topic['dominant_topic']==topic]
    with open(directory+'clusters/cluster_{}.txt'.format(topic), 'w') as f:
        for tweet in df['tweet']:
                f.write(str(tweet).strip()+'\n')