from __future__ import print_function
import numpy as np
from sklearn import decomposition
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
import re
import os
import pandas as pd
import json
from collections import defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

directory = os.getcwd()

# stop words
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
custom_stop_words = open(directory+"/stop_words.txt", "r" ).readlines()
custom_stop_words = [w.strip() for w in custom_stop_words]
stop_words.extend(custom_stop_words)
print("Stopword list has %d entries" % len(custom_stop_words) )

def remove_duplicates(lst):
    return list(dict.fromkeys(lst))

def lemmatize_sent(sent):
    word_list = nltk.word_tokenize(sent)
    return ' '.join([lemmatizer.lemmatize(w) for w in word_list])

# import data
directory = os.getcwd()
for i in range(1,8,1):
    data = pd.read_csv(directory+'/metoo_tweets_{}.csv'.format(i))
    if i==1:
        raw_documents = list(data['text'])
    else:
        raw_documents.extend(list(data['text']))

raw_documents = [re.sub(r"<.*?>", "", d) for d in raw_documents]
raw_documents = [re.sub("me_too", "", d) for d in raw_documents]
raw_documents = [re.sub("|", "", d) for d in raw_documents]
raw_documents = [d.strip() for d in raw_documents if d.strip()]
raw_documents_copy = raw_documents
raw_documents = [lemmatize_sent(d) for d in raw_documents if d.strip()]
lemma_dict = dict(zip(raw_documents, raw_documents_copy))
with open(directory+'/lemma_dict.json', 'w') as f:
    json.dump(lemma_dict, f, indent=2)

raw_documents = list(filter(None, raw_documents))
raw_documents = list(dict.fromkeys(raw_documents))
print("Imported %d documents" % len(raw_documents))

# tf-idf vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, min_df=20, max_df=0.9)
A = vectorizer.fit_transform(raw_documents)
print( "Created %d X %d TF-IDF-normalized document-term matrix" % (A.shape[0], A.shape[1]) )

# extract the resulting vocabulary
terms = vectorizer.get_feature_names()
print("Vocabulary has %d distinct terms" % len(terms))

# top ranking terms
import operator
def rank_terms( A, terms ):
    # get the sums over each column
    sums = A.sum(axis=0)
    # map weights to the terms
    weights = {}
    for col, term in enumerate(terms):
        weights[term] = sums[0,col]
    # rank the terms by their weight over all documents
    return sorted(weights.items(), key=operator.itemgetter(1), reverse=True)

ranking = rank_terms( A, terms )
for i, pair in enumerate( ranking[0:20] ):
    print( "%02d. %s (%.2f)" % ( i+1, pair[0], pair[1] ) )

# train the models
from sklearn import decomposition
k_min, k_max = 5, 70
step = 5
topic_models = []

import pyLDAvis.sklearn
for k in range(k_min, k_max+1, step):
    print("Applying topic modeling for k=%d ..." % k)
    # model = decomposition.NMF(init="nndsvd", n_components=k)
    model = decomposition.LatentDirichletAllocation(n_components=k, max_iter=100, learning_method='online')
    W = model.fit_transform(A)
    H = model.components_    
    # store for later
    topic_models.append((k,W,H))

    panel = pyLDAvis.sklearn.prepare(model, A, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, directory+'/vis/lda_{}.html'.format(k))

# build word embedding
class TokenGenerator:
    def __init__( self, documents, stopwords ):
        self.documents = documents
        self.stopwords = stopwords
        self.tokenizer = re.compile( r"(?u)\b\w\w+\b" )

    def __iter__( self ):
        print("Building Word2Vec model ...")
        for doc in self.documents:
            tokens = []
            for tok in self.tokenizer.findall( doc ):
                if tok in self.stopwords:
                    tokens.append( "<stopword>" )
                elif len(tok) >= 2:
                    tokens.append( tok )
            yield tokens

import gensim
docgen = TokenGenerator( raw_documents, custom_stop_words )
# the model has 500 dimensions, the minimum document-term frequency is 20
w2v_model = gensim.models.Word2Vec(docgen, size=500, min_count=20, sg=1)
print("Model has %d terms" % len(w2v_model.wv.vocab))

# caculate coherence score
def calculate_coherence( w2v_model, term_rankings ):
    overall_coherence = 0.0
    for topic_index in range(len(term_rankings)):
        # check each pair of terms
        pair_scores = []
        for pair in combinations(term_rankings[topic_index], 2):
            pair_scores.append(w2v_model.wv.similarity(pair[0], pair[1]))
        # get the mean for all pairs in this topic
        topic_score = sum(pair_scores) / len(pair_scores)
        overall_coherence += topic_score
    # get the mean score across all topics
    return overall_coherence / len(term_rankings)

import numpy as np
from itertools import combinations
def get_descriptor( all_terms, H, topic_index, top ):
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( all_terms[term_index] )
    return top_terms

k_values = []
coherences = []
for (k,W,H) in topic_models:
    # Get all of the topic descriptors - the term_rankings, based on top 10 terms
    term_rankings = []
    for topic_index in range(k):
        term_rankings.append(get_descriptor( terms, H, topic_index, 10 ))
    # Now calculate the coherence based on our Word2vec model
    k_values.append( k )
    coherences.append( calculate_coherence(w2v_model, term_rankings))
    print("K=%02d: Coherence=%.4f" % ( k, coherences[-1] ) )

# plot coherence scores
plt.style.use("ggplot")
matplotlib.rcParams.update({"font.size": 14})
fig = plt.figure(figsize=(13,7))
# create the line plot
ax = plt.plot( k_values, coherences )
plt.xticks(k_values)
plt.xlabel("Number of Topics")
plt.ylabel("Mean Coherence")
# add the points
plt.scatter( k_values, coherences, s=120)
# find and annotate the maximum point on the plot
ymax = max(coherences)
xpos = coherences.index(ymax)
best_k = k_values[xpos]
plt.annotate( "k=%d" % best_k, xy=(best_k, ymax), xytext=(best_k, ymax), textcoords="offset points", fontsize=16)
# show the plot
plt.show()

k_best = int(input("Choose k_best: "))
k = k_best
# get the model that we generated earlier.
W = topic_models[(k-k_min)//step][1]
H = topic_models[(k-k_min)//step][2]

for topic_index in range(k):
    descriptor = get_descriptor(terms, H, topic_index, 20)
    str_descriptor = ", ".join(descriptor)
    print("Topic %02d: %s" % (topic_index+1, str_descriptor))

def plot_top_term_weights( terms, H, topic_index, top ):
    # get the top terms and their weights
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    top_terms = []
    top_weights = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
        top_weights.append( H[topic_index,term_index] )
    # note we reverse the ordering for the plot
    top_terms.reverse()
    top_weights.reverse()
    # create the plot
    fig = plt.figure(figsize=(13,8))
    # add the horizontal bar chart
    ypos = np.arange(top)
    ax = plt.barh(ypos, top_weights, align="center", color="green",tick_label=top_terms)
    plt.xlabel("Term Weight",fontsize=14)
    plt.tight_layout()
    plt.show()

# plot_top_term_weights(terms, H, 0, 20)

def get_top_documents(all_documents, W, topic_index, top):
    # reverse sort the values to sort the indices
    top_indices = np.argsort(W[:,topic_index])[::-1]
    # now get the documents corresponding to the top-ranked indices
    top_documents = []
    for doc_index in top_indices[0:top]:
        top_documents.append(all_documents[doc_index])
    return top_documents

topic_documents = get_top_documents(raw_documents, W, 0, 10)
for i, document in enumerate(topic_documents):
    print("%02d. %s" % ((i+1), document))

def get_top_document_topic(all_documents, W, document_index):
    # reverse sort the values to sort the indices
    top_indices = np.argsort(W[document_index,:])[::-1]
    # get the documents corresponding to the top-ranked indices
    top_topic = top_indices[0]
    return top_topic

document_topic = dict()
for document in range(len(raw_documents)):
    document_topic[raw_documents[document]] = get_top_document_topic(raw_documents, W, document)

from spacy.lang.en import English
sentencizer = English()
sentencizer.add_pipe(sentencizer.create_pipe("sentencizer"))

doc_count = defaultdict()
for topic in range(k):
    doc_count[topic] = 0
    with open(directory+"/clusters/topic_{}.txt".format(topic), "w") as f:
        for d, t in document_topic.items():
            if t == topic:
                doc = sentencizer(d)
                sents = list(doc.sents)
                for i in range(len(sents)):
                    tag = 'T{}_D{}_S{}'.format(topic,doc_count[topic],i)
                    f.write(tag+'|'+lemma_dict[str(sents[i])]+'\n')
                doc_count[topic] += 1
print(doc_count)
                # d_utf = d.decode("utf-8")
                # d_ascii = d_utf.encode("ascii", "ignore")
                # f.write((d.decode("utf-8")).encode("ascii", "ignore")+"\n")