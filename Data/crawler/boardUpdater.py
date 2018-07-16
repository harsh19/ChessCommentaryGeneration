import pickle
import sys
import math
import copy
from collections import defaultdict


#To be rolled into class constructor
startState=[]
initDict={}
initDict['black_rook']=['a8','h8']
initDict['black_knight']=['b8','g8']
initDict['black_bishop']=['c8','f8']
initDict['black_queen']=['d8']
initDict['black_king']=['e8']
initDict['black_pawn']=['a7','b7','c7','d7','e7','f7','g7','h7']
initDict['white_rook']=['a1','h1']
initDict['white_knight']=['b1','g1']
initDict['white_bishop']=['c1','f1']
initDict['white_queen']=['d1']
initDict['white_king']=['e1']
initDict['white_pawn']=['a2','b2','c2','d2','e2','f2','g2','h2']
initDict[""]=[]
for row in ['a','b','c','d','e','f','g','h']:
    for col in range(3,7):
        initDict[""].append(row+str(col))


for key,val in initDict.items():
    for valElem in val:
        newDict={}
        if key!="":
            newDict['piece']=key
        newDict['location']=valElem
        startState.append(newDict)

startState.sort(key = lambda x:x['location'])


pieceCodes={}
pieceCodes[""]="pawn"
pieceCodes["B"]="bishop"
pieceCodes["K"]="king"
pieceCodes["N"]="knight"
pieceCodes["Q"]="queen"
pieceCodes["R"]="rook"

revPieceCodes={}
for key,val in pieceCodes.items():
    revPieceCodes[val]=key

def getIndex(positionString):
    firstChar=positionString[0]
    secondChar=positionString[1]
    row=ord(firstChar)-ord('a')
    col=ord(secondChar)-ord('1')
    return (row*8+col)

def rowIncrement(start,K):
    index=start+8*K
    if index<0 or index>=64:
        index=-1
    return index

def colIncrement(start,K):
    index=index+K
    if index<0 or index>=64:
        index=-1
    return index

def parseMoveHeader(moveString):
    words=moveString.split()
    
    counter=0
    if "..." in words[0]:
        firstMove="black"
    else:
        firstMove="white"

    moves=[]
    for i,word in enumerate(words):
        if "." in word:
            continue
        elif "..." in word:
            continue
        else:
            moves.append((word,firstMove))
            if firstMove=="black":
                firstMove="white"
            else:
                firstMove="black"

    return moves

def copyState(boardState):
    copiedState=[]
    for elem in boardState:
        newElem={}
        newElem=copy.deepcopy(elem)
        copiedState.append(newElem)
    return copiedState

def reverseIndex(boardState):
    boardIndex={}
    for elem in boardState:
        if 'piece' in elem:
            piece=elem['piece']
            location=elem['location']
            if piece not in boardIndex:
                boardIndex[piece]=[]
            boardIndex[piece].append(location)
    return boardIndex

def movePawn(boardState,boardIndex,moveColor,destination):
    if moveColor=="white":
        destinationRow=destination[0]
        destinationCol=destination[1]
        #There is only one pawn of any color for a given letter
        startPoints=boardIndex['white_pawn']
        if len(startPoints)==1:
            startPoint=startPoints[0]
        else:
            for startPoint in startPoints:
                startRow=startPoint[0]
                startCol=startPoint[1]



#def moveKing():

#def moveBishop():

#def moveKing():

#def moveQueen():

#def moveRook():

#def capturePawn():

#def kingsCastle():

#def QueensCastle():

#def enPassant():

def updateState(boardState,boardIndex,move):
    #Updates both boardState and boardIndex
    moveColor=move[1]
    moveString=move[0]
    

    if "x" in move:
        return
    elif "-" in move:
        return
    elif "#" in move:
        return
    else:
        if len(moveString)==2:
            destination=moveString
            movePawn(boardState,boardIndex,moveColor,destination)
        else:
            return



if __name__=="__main__":
    all_links=pickle.load(open("./saved_files/saved_links.p","r"))
    extra_links=pickle.load(open("extra_pages.p","r"))

    for i,link in enumerate(all_links):
       
        pageLength=0
        if extra_links[i]>0:
            pageLength=1+(extra_links[i]-1)
        else:
            pageLength=1

        currentState=copyState(startState)
        currentIndex=reverseIndex(currentState)
        for pageNo in range(pageLength):
            if pageNo==0:
                pageObjName="./outputs/saved"+str(i)+".obj"
            else:
                pageObjName="./outputs/saved"+str(i)+"_"+str(pageNo)+".obj"
            
            data=None
            try:
                #This is one page
                data=pickle.load(open(pageObjName,"r"))
            except:
                break
            print len(data)
            for elem in data:
                moveSequence=elem[0]
                boardState=elem[1]
                comment=elem[2]
                parsedMoveSequence=parseMoveHeader(moveSequence)
                print moveSequence
                print parsedMoveSequence
                print boardState
                currentState=boardState
                currentIndex=reverseIndex(boardState)
                print currentIndex
                break

        break


