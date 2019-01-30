import pickle
# coding: utf-8

# In[1]:

import sys
from convertToFEN import convertToFEN
#from boardInfo import BoardInfo
import chess
import chess.uci
import numpy as np


'''
def getData(split, src, src_dir="./data/"):
    che_data = open(src_dir + split + src + ".che", "r").readlines()
    en_data = open(src_dir + split + src + ".en", "r").readlines()
    return che_data, en_data
'''

info_handler = None #chess.uci.InfoHandler()

engine = chess.uci.popen_engine("/usr/games/stockfish")
engine.uci()
info_handler = chess.uci.InfoHandler()
engine.info_handlers = [] #.clear()
engine.info_handlers.append(info_handler)


def getIndexFromRankFile(rank_file_string):
    file_to_val = {"a":0, "b":1, "c":2, "d":3, "e":4 , "f":5, "g":6, "h":7}
    file,rank = rank_file_string[0], rank_file_string[1]
    file = file_to_val[file]
    rank = int(rank)-1
    index = 8*rank+file
    return index

def getRankFileFromIndex(index):
    file_to_val = {"a":0, "b":1, "c":2, "d":3, "e":4 , "f":5, "g":6, "h":7}
    val_to_file = {v:k for k,v in file_to_val.items()}
    rank = 1+index/8
    file = index%8
    rank = str(rank)
    file = val_to_file[file]
    return file+rank
    


# Score features

def getPieceCountFeatures(board):
    cnts_black = {'r':0,'k':0,'q':0,'n':0,'b':0,'p':0}
    cnts = {'R':0,'K':0,'Q':0,'N':0,'B':0,'P':0}
    for sq in chess.SQUARES:
        pi = board.piece_at(sq)
        #print type(pi)
        pi = str(pi)
        if pi is not None:
            if pi in cnts_black:
                cnts_black[pi]+=1
            if pi in cnts:
                cnts[pi]+=1
    vals = []
    for k,v in cnts.items():
        if k=='P':
            v/=8.0
        elif k!='K' and k!='Q':
            v/=2.0
        vals.append(v)
    #print "cnts_black.items() : ",cnts_black.items()
    for k,v in cnts_black.items():
        if k=='p':
            #print k
            v/=8.0
        elif k!='k' and k!='q':
            v/=2.0
        vals.append(v)
    
    #print vals
    return vals

def getScore(board):
    runs = 5
    vals = np.zeros(runs)
    for run in range(runs):
        engine.position(board)
        engine.go(depth=10) #movetime=5000)
        score = info_handler.info["score"][1]
        #print type(score)
        #print score
        score = score.cp
        vals[run] = score
    return [np.mean(vals), np.max(vals)]
    

def getScoreFeatures(board):
    score = getScore(board)
    if score>600:
        score = 600.0
    if score<-600:
        score = -600
    return [score/100.0]

def getScoreDiffFeatures(board, move_str):
    s1 = getScore(board)
    b = board #chess.board(board)
    mv = chess.Move.from_uci( move_str )
    b.push( mv )
    s2 = getScore(b)
    return [ ( -s2[0] - s1[0])/100.0, (-s2[1] - s1[1])/100.0] # , s1, -s2]
    

def getPawnFeatures(board):
    black = []
    white = []
    for sq in chess.SQUARES:
        pi = board.piece_at(sq)
        #print type(pi)
        pi = str(pi)
        if pi=="p":
            black.append(chess.square_rank(sq))
        elif pi=="P":
            white.append(chess.square_rank(sq))
    #print black
    #print white
    avg_b, avg_w, max_b, max_w = 0.0,0.0,0.0,0.0
    if len(black)>0:
        avg_b = 7.0 - np.mean(black)
        max_b = 7.0 - np.max(black)
    if len(white)>0:
        avg_w = np.mean(white)
        max_w = np.max(white)
    return [avg_b, avg_w, max_b, max_w]
    

def complement(string):
    if string=="black":
        return "white"
    else:
        return "black"

    
def getAllScoreFeatures(board, move_str):
    feats1 = getScoreFeatures(board)
    #print feats1
    feats2 = getPawnFeatures(board)
    #print feats2
    feats3 = getPieceCountFeatures(board)
    #print feats3
    feats4 = getScoreDiffFeatures(board, move_str)
    return feats1+feats2+feats3+feats4



def parseChessString(words):
    eop_index = words.index("<EOP>")
    eoc_index = words.index("<EOC>")
    current_board = words[:eoc_index]
    previous_board = words[eoc_index+1:eop_index]
    player_to_move = words[eop_index+1]
    player_to_move_next = complement(player_to_move)
    square_to_move = words[eop_index+4]
    square_to_moveTo = words[eop_index+5]
    move_color = 'w'
    move_colorNext = 'b'
    if player_to_move == "black":
        move_color="b"
        move_colorNext="w"
    return current_board, previous_board, square_to_move, square_to_moveTo, move_color, move_colorNext



def main(src_dir, src, typ):

    che_data = open(src_dir + src +".che", "r").readlines()
    en_data = open(src_dir + src +".en", "r").readlines()
    all_train_feats = []

    all_feats = []
    print "len(che_data) : ",len(che_data)
    i = 0
    for line in che_data:

        chess_str = line.split()
        current_board, previous_board, square_to_move, square_to_moveTo, move_color, move_colorNext = parseChessString(chess_str)
        
        cur_board_fen = convertToFEN(current_board, move_colorNext)
        prev_board_fen = convertToFEN(previous_board, move_color)
        #print cur_board_fen
        cur_board = chess.Board(cur_board_fen)
        prev_board = chess.Board(prev_board_fen)

    
        #attackers_list, attacks_list = getAttackers(cur_board_fen, player_to_move, square_to_moveTo)
        #board_score = getScoreFeatures(cur_board_fen)
        score_feats = None
        try:
            score_feats = getAllScoreFeatures(prev_board, square_to_move + square_to_moveTo)
        except:
            print "error occurred. all zero vectors"
            score_feats = [0]*19
        all_train_feats.append( score_feats )
        #prev_board = chess.Board(prev_board_fen) # board would have got changed
        print score_feats
        all_feats.append(score_feats)
        #print cur_board
        i+=1
        #break
        #if i>20:
        #    break
    
    pickle.dump( all_feats, open( "./feature_dumps/" + src + "."+typ+".all_feats.pickle","w") )

#main()