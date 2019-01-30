from sklearn.neighbors import NearestNeighbors
import numpy as np
import pickle
import random

import sys
import argparse
import copy

import logging

def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--use_score',dest='use_score',action='store_true',default=False) 
    parser.add_argument('--use_move',dest='use_move',action='store_true',default=False) 
    parser.add_argument('--use_threat',dest='use_threat',action='store_true',default=False) 
    parser.add_argument('-src',dest='src',default='che-eng.0') 
    args=parser.parse_args()
    return args


def getData(split, params):
    all_feats = []
    src = params.src
    if params.use_score:
        typ = "score"
        all_feats.append( pickle.load( open( "./feature_dumps/" + split + "."+src + "."+typ+".all_feats.pickle","r") ) )
    if params.use_move:
        typ="move"
        all_feats.append( pickle.load( open( "./feature_dumps/" + split + "."+src + "."+typ+".all_feats.pickle","r") ) )
    if params.use_threat:
        typ = "threat"
        all_feats.append( pickle.load( open( "./feature_dumps/" + split + "."+src + "."+typ+".all_feats.pickle","r") ) )
    all_feats = np.hstack(all_feats)
    #all_feats = np.array(all_feats)
    all_feats = np.nan_to_num(all_feats, copy=False)
    all_texts = open("./data/" + split+".che-eng.0.en", "r").readlines()
    #print "all_feats : ",all_feats.shape[0]
    #logging.info("all_feats :")
    #print "all_texts:", len(all_texts)
    assert all_feats.shape[0] == len(all_texts)
    return all_feats, all_texts


def getStringSim(s1, s2):
    #TODO
    return 0.0


def learnFunc(train_feats, train_texts, val_feats, val_texts):
    num_train = train_feats.shape[0]
    lim = 20
    sim_training_data = []
    for i in range(num_feats-1):
        indices = np.arange(i+1, num_feats, 1)
        np.random.shuffle(indices)
        for j in indices[:lim]:
            sim_val = getStringSim(train_texts[i], train_texts[j])
            sim_training_data.append( np.hstack([train_feats[i], train_feats[j] ]) )
            sim_training_labels.append(sim_val)


def main():

    params = parseArguments()

    #typ = sys.argv[1] #"score"
    #src = sys.argv[2] #"che-eng.0"
    #fw = open("outputs/" + typ + ".test.txt")

    learn_func = False
    train_feats, train_texts = getData('train', params)
    test_feats, test_texts = getData('test', params)
    num_feats = train_feats.shape[1]

    val_feats, val_texts = None, None
    weights = np.ones(num_feats)
    if learn_func:
        val_feats, val_texts = getData('val', params)
        weights = learnFunc(train_feats, train_texts, val_feats, val_texts)


    neighbors = 5
    nbrs = NearestNeighbors(n_neighbors=neighbors, algorithm='auto').fit(train_feats)
    #print nbrs
    for i,(feats, text) in enumerate( zip(test_feats, test_texts) ):
        test_points = test_feats[i:i+1]
        distances, indices = nbrs.kneighbors(test_points)
        #print distances, indices
        output = train_texts[indices[0][0]]
        print output.strip()
        #print text.strip()
        #print ""
        #if i>5:
        #    break


main()