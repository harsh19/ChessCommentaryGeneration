# coding: utf-8
import pickle

####


####



# In[1]:

import sys
from convertToFEN import convertToFEN
#from boardInfo import BoardInfo
import chess
import chess.uci
import numpy as np


####################################################### chess utils
'''
def getData(split, src, src_dir="./data/"):
    che_data = open(src_dir + split + src + ".che", "r").readlines()
    en_data = open(src_dir + split + src + ".en", "r").readlines()
    return che_data, en_data
'''
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
    

def complement(string):
    if string=="black":
        return "white"
    else:
        return "black"

def parseChessString(words):
    eop_index = words.index("<EOP>")
    eoc_index = words.index("<EOC>")
    current_board = words[:eoc_index]
    previous_board = words[eoc_index+1:eop_index]
    player_to_move = words[eop_index+1]
    player_to_move_next = complement(player_to_move)
    from_position = words[eop_index+4]
    to_position = words[eop_index+5]
    current_player = 'w'
    other_player = 'b'
    if player_to_move == "black":
        current_player="b"
        other_player="w"
    return current_board, previous_board, from_position, to_position, current_player, other_player




def getColor(piece):
    if piece.islower():
        return 'b'
    else:
        return 'w'
    
    
def getPiecesAtPositions(board, position_list, player):
    ret = []
    for position in position_list:
        piece = board.piece_at(position)
        if piece is not None:
            #print getRankFileFromIndex(position), piece.symbol(), getColor(piece.symbol()), player
            if getColor(piece.symbol()) == player:
                ret.append(piece.symbol())
    return ret


###############
PIECE_TYPES = [NOTHING, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(0, 7)
PIECE_SYMBOLS = ["", "p", "n", "b", "r", "q", "k"]
PIECE_NAMES = ["nothing", "pawn", "knight", "bishop", "rook", "queen", "king"]
pice_to_one_hot_dct = { k:v for k,v in zip(PIECE_NAMES,range(0,7)) }

def getMoveNamesToFeatures(piece_str):
    ret = [0]*7
    piece_str = piece_str.lower()
    ret[ pice_to_one_hot_dct[piece_str] ] = 1
    return ret
#driver
#pieces =  ['B']
#print getMoveNamesToFeatures(pieces)

#######################################################

def getMoveRep(data_point):
    vals = data_point.split("<EOP> ")[1]
    vals = vals.split(" <EOM>")[0]
    vals = vals.split()[1:]
    return vals



##############################################
def getMoveFeatures(chess_str):

    ret = []
    
    vals = getMoveRep(chess_str)
    
    # piece
    piece = vals[1]
    #print "piece = ", piece
    #ret.extend( getMoveNamesToFeatures(piece) ) 
    ret.append('move_'+piece)
    
    # color
    piece_color = vals[0] 
    ret.append('movecolor_'+piece_color)
    '''
    if piece_color == "black":
        ret.append(0)
    else:
        ret.append(1)
    #print "piece_color = ", piece_color, ret
    '''

    piece_from = vals[2]
    idx = -1
    try:
        idx = getIndexFromRankFile(piece_from)
    except:
        print "error : piece_from = ", piece_from
    ret.append('movefrom_'+piece_from)
    #tmp = [0]*64
    #if idx!=-1:
    #    tmp[idx] = 1
    #ret.extend(tmp)
    #print "piece from feats = ", tmp, piece_from, idx

    piece_to = vals[3]
    ret.append('moveto_'+piece_to)

    
    capture_color = "none"
    capture_piece = "none"
    tmp = []
    if "capture" in vals:
        tmp.append(1) # 1
        idx = vals.index("capture")
        try:
            capture_color = vals[idx+1]
            if capture_color=="black": 
                tmp.extend([0,1]) # 2
            elif capture_color == "white":
                tmp.extend([1,0])
            capture_piece = vals[idx+2]
            tmp.extend( getMoveNamesToFeatures(capture_piece) ) # 7
        except:
            print "error for vals = ", vals
            tmp = [0] * 10
    else:
        tmp = [0] * 10
    #print "capt feats = ", tmp
    ret.append("capturecolor_"+capture_color)
    ret.append("capturepiece_"+capture_piece)

    #ret.extend(tmp)

    if 'castling' in vals:
        ret.append("castling_true")
    else:
        ret.append("castling_false")

    if 'check' in vals:
        ret.append("check_true")
    else:
        ret.append("check_false")

    #print "ret = ", ret

    return ret

##############################################

def main(src_dir, src, typ="move"):

    che_data = open(src_dir + src +".che", "r").readlines()
    en_data = open(src_dir + src +".en", "r").readlines()
    all_train_feats = []

    all_feats = []
    print "len(che_data) : ",len(che_data)
    i = 0
    for line in che_data:
        #print "i= ",i
        chess_str = line.strip() #.split()
        move_features = getMoveFeatures( chess_str )        
        all_train_feats.append( move_features )
        print "move_features = ", move_features
        assert len(move_features)==8
        all_feats.append(move_features)
        i+=1
        #break
        #if i>93:
        #    break
        #print "="*99

    print "all_feats: ", len(all_feats), len(all_feats[0])
    #print np.array(all_feats).shape
    #print  "./feature_dumps/" + src + "."+typ+".all_feats.pickle"
    pickle.dump( all_feats, open( "./feature_dumps/" + src + "."+typ+".sequences.pickle","w") )

#main()