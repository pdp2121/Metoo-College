import os
import json
import sys
from nltk.corpus import wordnet as wn
from collections import defaultdict
import gensim.downloader as api
from nltk.stem.wordnet import WordNetLemmatizer
import random

directory = os.getcwd()
lemmatizer = WordNetLemmatizer()
glove = api.load("glove-wiki-gigaword-300")
with open(directory+'/vocab.txt', 'w') as f:
    f.write(str(glove.vocab.keys()))
print('Vocab has {} terms'.format(len(glove.vocab.keys())))

frame_file = open(directory+'/frames.txt', 'r')
frame = dict()
tags = ['perspective|writer-object', 
        'perspective|writer-subject', 
        'perspective|subject-object',
        'verb-sentiment|object',
        'verb-sentiment|subject',
        'entity-importance|object',
        'entity-importance|subject',
        'entity-sentiment|object',
        'entity-sentiment|subject',
        'perspective|reader-object', 
        'perspective|reader-subject', 
        'perspective|object-subject']

for line in frame_file.readlines()[1:]:
    line = line.strip()
    verb = line.split('\t')[0]
    frame[verb] = dict()
    for i in range(len(tags)):
        tag = tags[i].split('|')
        field_tag = tag[0]
        sub_tag = tag[1]
        if field_tag not in frame[verb].keys():
            frame[verb][field_tag] = dict()
            frame[verb][field_tag][sub_tag] = line.split('\t')[i+1]
        else:
            frame[verb][field_tag][sub_tag] = line.split('\t')[i+1]

# frame_out = json.dumps(frame, indent=2)
# with open(directory+'/frame.json', 'w') as f:
#     f.write(frame_out)

def generate_score(word):
    if word not in glove.vocab.keys():
        return None

    glove_neighbors = glove.most_similar(word)
    glove_neighbors = [(w,p) for (w,p) in glove_neighbors if w in frame.keys()]
    # print(glove_neighbors)
    norm = sum([p for (w,p) in glove_neighbors])
    # norm = len(glove_neighbors)
    
    if len(glove_neighbors)>0:
        frame[word] = dict()
        for i in range(len(tags)):
            tag = tags[i].split('|')
            field_tag = tag[0]
            sub_tag = tag[1]
            if field_tag not in frame[word].keys():
                frame[word][field_tag] = dict()
                frame[word][field_tag][sub_tag] = str(float(sum([float(frame[w][field_tag][sub_tag])*p/norm for (w,p) in glove_neighbors])))
            else:
                frame[word][field_tag][sub_tag] = str(float(sum([float(frame[w][field_tag][sub_tag])*p/norm for (w,p) in glove_neighbors])))
        score = frame[word]['verb-sentiment']['object']

        frame_out = json.dumps(frame, indent=2)
        with open(directory+'/frame.json', 'w') as f:
            f.write(frame_out)
    else:
        score = None
    return score


def extract_features(node, tree, parent=None, scoring=None, agent=True):
    features = dict()

    word = tree[node]['word']
    ont_type = tree[node]['type']

    if word is not None:
        word = word.lower()
    if ont_type is not None:
        ont_type = ont_type.lower()

    features['word'] = word
    features['ont_type'] = ont_type

    if scoring == 'predicate' and word is not None:
        if len(word.split('-'))>1:
            word = word.split('-')[0]

        if word == 'raped':
            word = 'rape'
        else:
            word = lemmatizer.lemmatize(word, 'v')
        # print(word)

        if word in frame.keys():
            features['verb-sentiment'] = frame[word]['verb-sentiment']['object']
            features['reader-agent-sentiment'] = frame[word]['perspective']['reader-subject']
            features['reader-affected-sentiment'] = frame[word]['perspective']['reader-object']
            features['affected-agent-sentiment'] = frame[word]['perspective']['object-subject']
        else:
            features['verb-sentiment'] = generate_score(word)
            if features['verb-sentiment'] is not None:
                features['reader-agent-sentiment'] = frame[word]['perspective']['reader-subject']
                features['reader-affected-sentiment'] = frame[word]['perspective']['reader-object']
                features['affected-agent-sentiment'] = frame[word]['perspective']['object-subject']

    if scoring == 'entity' and word is not None:
        if parent in frame.keys():
            if agent == True:
                features['entity-sentiment'] = frame[parent]['entity-sentiment']['subject']
                features['entity-importance'] = frame[parent]['entity-importance']['subject']
            else:
                features['entity-sentiment'] = frame[parent]['entity-sentiment']['object']
                features['entity-importance'] = frame[parent]['entity-importance']['object']
    
    return features

topic = 'topic_1'

sent_dict = dict()
harr_cat_dict = dict()
parti_cat_dict = dict()
sent_file = open(directory+'/'+topic+'.txt', 'r')
for line in sent_file.readlines():
    line = line.strip()
    id_tag = line.split('|')[0]
    sent = line.split('|')[1]
    sent_dict[id_tag] = sent

    # print(id_tag)
    if len(line.split('|'))>2:
        harr_cat_dict[id_tag] = line.split('|')[2]
        parti_cat_dict[id_tag] = line.split('|')[3]
    else:
        harr_cat_dict[id_tag] = ""
        parti_cat_dict[id_tag] = ""

    

total_features = dict()

id_tags = [str(k) for k in sent_dict.keys()]
for iters in range(5):
    for fn in random.sample(id_tags, k=len(id_tags)):
        file_name = '{}.json'.format(fn)
        if os.stat(directory+'/'+topic+'/'+file_name).st_size != 0:
            # print(file_name)
            trips_dict = json.load(open(directory+'/'+topic+'/'+file_name, 'r'))
            if (trips_dict is not None) and (isinstance(trips_dict, dict)):
                trips_tree = trips_dict['parse'][0]

                if trips_tree is not None:
                    feature_dict = dict()
                    feature_dict['id'] = file_name.split('.json')[0]
                    feature_dict['sentence'] = sent_dict[file_name.split('.json')[0]]
                    feature_dict['harassment_label'] = harr_cat_dict[file_name.split('.json')[0]]
                    feature_dict['paticipant_label'] = parti_cat_dict[file_name.split('.json')[0]]

                    verb_dict = dict()
                    for node in trips_tree.keys():
                        if node != 'root':
                            # print(node)
                            # print(trips_tree[node]['roles'].keys())
                            if trips_tree[node]['word'] is not None:
                                word = trips_tree[node]['word'].lower()
                            if 'AGENT' in trips_tree[node]['roles'].keys() or 'AFFECTED' in trips_tree[node]['roles'].keys():
                                verb_dict[word] = extract_features(node, trips_tree, scoring='predicate')
                                if 'AGENT' in trips_tree[node]['roles'].keys() and trips_tree[node]['roles']['AGENT'][1:] in trips_tree.keys():
                                    verb_dict[word]['agent'] = extract_features(trips_tree[node]['roles']['AGENT'][1:], trips_tree, scoring='entity', parent=word)
                                if 'AFFECTED' in trips_tree[node]['roles'].keys() and trips_tree[node]['roles']['AFFECTED'][1:] in trips_tree.keys():
                                    verb_dict[word]['affected'] = extract_features(trips_tree[node]['roles']['AFFECTED'][1:], trips_tree, scoring='entity', parent=word, agent=False)
                            
                    feature_dict['predicates'] = verb_dict
                    # if len(feature_dict['predicates'])>0:

                    if feature_dict['id'] not in total_features.keys():
                        total_features[fn] = feature_dict
                    else:
                        if feature_dict != total_features[fn]:
                            print('Update feature:', fn)
                            total_features[fn] = feature_dict

    print('Connotation Frames has {} instances'.format(len(frame)))

frame_out = json.dumps(frame, indent=2)
with open(directory+'/frame.json', 'w') as f:
    f.write(frame_out)

feature_out = json.dumps(list(total_features.values()), indent=2)
with open(directory+'/feature.json', 'w') as f:
    f.write(feature_out)