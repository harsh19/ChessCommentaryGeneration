import pickle
# coding: utf-8

# In[1]:

import sys
from convertToFEN import convertToFEN
import chess
import chess.uci
import numpy as np
import scipy.stats

def main():
	dir = "./feature_dumps/"
	split = "train"
	data_typ = "che-eng.0"
	feat_typ = "score"
	feats = pickle.load( open(dir+split+"."+data_typ+"."+feat_typ + ".all_feats.pickle","r") )
	feats = np.array(feats)
	print feats.shape    
	np.nan_to_num(feats, copy=False)
	for j in range(feats.shape[1]):
		print "j = 1",j
		feats_j = feats[:,j]
		print scipy.stats.describe(feats_j)

main()