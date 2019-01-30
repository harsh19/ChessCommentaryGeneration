import pickle
# coding: utf-8

# In[1]:

import sys
from convertToFEN import convertToFEN
import chess
import chess.uci
import numpy as np


'''
def getData(split, src, src_dir="./data/"):
    che_data = open(src_dir + split + src + ".che", "r").readlines()
    en_data = open(src_dir + split + src + ".en", "r").readlines()
    return che_data, en_data
'''
def getIndexFromRankFile(rank_file_string):
    file_to_val = {"a":0, "b":1, "c":2, "d":3, "e":4 , "f":5, "g":6, "h":7}
    file,rank = rank_file_string[0], rank_file_string[1]
    try:
        file = file_to_val[file]
    except:
        print "error: ",file," not found"
        file = 7
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


def getAttackers(board, player_to_move, to_square):
    if player_to_move == 'w':
        player_to_move = chess.WHITE
    else:
        player_to_move = chess.BLACK
    attackers = board.attackers(player_to_move, getIndexFromRankFile(to_square))
    attacks = board.attacks(getIndexFromRankFile(to_square))
    attackers_list = []
    attacks_list = []
    for attacker in attackers:
        attackers_list.append(attacker)
    for attacker in attacks:
        attacks_list.append(attacker)
    return attackers_list, attacks_list

def fnc(lst):
    return [getRankFileFromIndex(val) for val in lst]


def getAttackCountFeatures(board, attack_team_color=chess.BLACK):
    print "sdd"
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        print getRankFileFromIndex(sq), piece, getIndexFromRankFile( getRankFileFromIndex(sq) )
        if board.is_attacked_by( attack_team_color, sq):
            if piece is not None:
                pass #print getRankFileFromIndex(sq), piece
        
    #board.piece_at(chess.C5)
    #board.is_attacked_by(chess.WHITE, chess.E8)
    
    
###############
PIECE_TYPES = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING] = range(0, 6)
PIECE_SYMBOLS = ["p", "n", "b", "r", "q", "k"]
PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"]

def getMoveNamesToFeatures(lst):
    dct = { k:v for k,v in zip(PIECE_SYMBOLS,range(0,6)) }
    ret = [0]*6
    for piece_str in lst:
        piece_str = piece_str.lower()
        ret[ dct[piece_str] ] = 1
    return ret
#driver
#pieces =  ['B']
#print getMoveNamesToFeatures(pieces)

    

def getThreatFeatures(board, prev_board, current_player, other_player, from_position, to_position):
    
    ret = []

    # analyze threats on the piece before the threat
    attackers_list, attacks_list = getAttackers(prev_board, other_player, from_position)
    #print "attackers_list = ", fnc(attackers_list)
    pieces = getPiecesAtPositions(board, attackers_list, other_player)
    #print  "pieces = ", pieces
    ret.extend( getMoveNamesToFeatures(pieces) )

    # analyze threats by the piece after the threat
    attackers_list, attacks_list = getAttackers(board, current_player, to_position)
    #print "attacks_list = ", fnc(attacks_list)
    pieces = getPiecesAtPositions(board, attacks_list, other_player)
    #print  "pieces = ", pieces
    ret.extend( getMoveNamesToFeatures(pieces) )

    # analyze threats on the piece after the threat
    attackers_list, attacks_list = getAttackers(board, other_player, to_position)
    #print "attackers_list = ", fnc(attackers_list)
    pieces = getPiecesAtPositions(board, attackers_list, other_player)
    #print  "pieces = ", pieces
    ret.extend( getMoveNamesToFeatures(pieces) )

    return ret


def main(src_dir, src, typ="threat"):

    che_data = open(src_dir + src +".che", "r").readlines()
    en_data = open(src_dir + src +".en", "r").readlines()
    all_train_feats = []

    all_feats = []
    print "len(che_data) : ",len(che_data)
    i = 0
    for line in che_data:

        chess_str = line.split()
        current_board, previous_board, from_position, to_position, current_player, other_player = parseChessString(chess_str)
        
        cur_board_fen = convertToFEN(current_board, other_player)
        prev_board_fen = convertToFEN(previous_board, current_player)
        #print cur_board_fen
        cur_board = chess.Board(cur_board_fen)
        prev_board = chess.Board(prev_board_fen)

    
        threat_feats = getThreatFeatures(cur_board, prev_board, current_player, other_player, from_position, to_position)
        
        all_train_feats.append( threat_feats )
        print threat_feats
        all_feats.append(threat_feats)
        i+=1
        #if i>20:
        #    break
    
    pickle.dump( all_feats, open( "./feature_dumps/" + src + "."+typ+".all_feats.pickle","w") )

#main()