import pickle
# coding: utf-8

# In[1]:

import sys
from convertToFEN import convertToFEN
import chess
import chess.uci
import numpy as np

import generate_score_features
import generate_threat_features
import generate_move_features


def main():

    src_dir = sys.argv[1] # ./data/  ##data directory
    src = sys.argv[2] # "train.che-eng.0"  ## data file name
    #typ = sys.argv[3] # score ## save using this name

    generate_score_features.main(src_dir, src, "score")
    generate_threat_features.main(src_dir, src, "threat")
    generate_move_features.main(src_dir, src, "move")

main()