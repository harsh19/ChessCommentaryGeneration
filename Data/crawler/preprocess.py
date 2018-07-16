import pickle
import sys
import math
from collections import defaultdict
import textUtils
from nltk.tokenize import word_tokenize

def diffString(x,y):
    boardChanges=[]
    
    for (previous,current) in zip(x,y):
        if 'piece' not in previous and 'piece' not in current:
            boardChanges.append('eps')
        if 'piece' in previous and 'piece' not in current:
            boardChanges.append('-'+previous['piece'])
        if 'piece' not in previous and 'piece' in current:
            boardChanges.append('+'+current['piece'])
        if 'piece' in previous and 'piece' in current:
            if previous['piece']==current['piece']:
                boardChanges.append('eps')
            else:
                boardChanges.append('+'+current['piece'])

    return " ".join(boardChanges)

def mapName(x):
    if x=="K":
        return "king"
    elif x=="Q":
        return "queen"
    elif x=="R":
        return "rook"
    elif x=="B":
        return "bishop"
    elif x=="N":
        return "knight"
    else:
        return "pawn "+x

def parseMove(move):
    if move[-1]=="+" or move[-1]=="#":
        move=move[:-1]

    if "x" not in move and len(move)==2:
        return "_pawn"+" "+move
    elif "x" not in move and len(move)==3:
        return "_"+mapName(move[0])+" "+move[1:]
    elif "x" in move:
        return "_"+mapName(move[0])+" "+"X"+" "+move[2:]
    else:
        return "_"+"<strangeMove>"


def parseMoveString(x):
    x=x.split()
    moveSequence=[]
    startIndex=0 
    if "..." in x[0]:
        x=x[1:]
        startIndex=2

    for elem in x:
        if startIndex%3!=0:
            if startIndex%3==1:
                moveSequence.append("white"+parseMove(elem)+" <EOM>")
            else:
                moveSequence.append("black"+parseMove(elem)+" <EOM>")
        startIndex+=1

    return moveSequence

if sys.argv[1]=="train":
    all_links=pickle.load(open("./saved_files/train_links.p","rb"))
elif sys.argv[1]=="valid":
    all_links=pickle.load(open("./saved_files/valid_links.p","rb"))
else:
    all_links=pickle.load(open("./saved_files/test_links.p","rb"))

outFileMultiChe=open("./saved_files/"+sys.argv[1]+".che-eng.multi.che","w")
outFileMultiEn=open("./saved_files/"+sys.argv[1]+".che-eng.multi.en","w")
outFileSingleChe=open("./saved_files/"+sys.argv[1]+".che-eng.single.che","w")
outFileSingleEn=open("./saved_files/"+sys.argv[1]+".che-eng.single.en","w")

outFileMultiCheStrings,outFileMultiEnStrings,outFileSingleCheStrings,outFileSingleEnStrings=[],[],[],[]

print all_links[0]

extra_links=pickle.load(open("extra_pages.p","r"))

moreThanOneLong=0
total=0
totalPages=0.0
averagePages=0.0
totalMoves=0.0
longMoves=0.0
averageMoves=0.0
wids=defaultdict(int)
word_frequencies=defaultdict(int)
word_game_frequencies={}
freqThreshold=3
totalTokens=0.0

for i,link in all_links:
    total+=1
    
    pageLength=0
    if extra_links[i]>0:
        pageLength=1+(extra_links[i]-1)
    else:
        pageLength=1

    if pageLength>1:
        moreThanOneLong+=1
    
    totalPages+=pageLength

    startState=pickle.load(open("outputs/startState.obj","rb"))

    for pageNo in range(pageLength):
        if pageNo==0:
            pageObjName="./outputs/saved"+str(i)+".obj"
        else:
            pageObjName="./outputs/saved"+str(i)+"_"+str(pageNo)+".obj"
        print pageObjName
        data=None
        try:
            #This is one page
            data=pickle.load(open(pageObjName,"r"))
            totalMoves+=len(data)
        except:
            break
           
        for elem in data:
            moves=elem[0]
            board=elem[1]
            comment=elem[2].encode('ascii','replace').strip()
            commentWords=" ".join(word_tokenize(comment))
            currentStateString=" ".join([x['piece'] if 'piece' in x else 'eps' for x in elem[1]])
            startStateString=" ".join([x['piece'] if 'piece' in x else 'eps' for x in startState[1]])
            diffStateString=diffString(startState[1],elem[1])
            print "Comment:",commentWords
            print "Start State:\n"+startStateString
            print "Moves:",moves
            moveSequence=parseMoveString(moves)
            print "Parsed Moves:",moveSequence
            print "Current State:\n"+currentStateString
            print "Diff State:\n"+diffStateString
            srcString=currentStateString+" <EOC> "+startStateString+" <EOP> "+" ".join(moveSequence)+" <EOMH>"
            tgtString=commentWords
            if len(moveSequence)==1:
                outFileSingleCheStrings.append(srcString+"\n")
                outFileSingleEnStrings.append(tgtString+"\n")
            
            outFileMultiCheStrings.append(srcString+"\n")
            outFileMultiEnStrings.append(tgtString+"\n")

            startState=elem


#Would have done random shuffles here but we are shuflling during training anyway
#outFileMultiTuples=[(outFileMultiCheStrings[i],outFileMultiEnStrings[i] for i in range(len(outFileMultiCheStrings))]
#outFileSingleTuples=[(outFileSingleCheStrings[i],outFileSingleEnStrings[i] for i in range(len(outFileSingleCheStrings))]
 

for line in outFileSingleCheStrings:
    outFileSingleChe.write(line)
outFileSingleChe.close()

for line in outFileSingleEnStrings:
    outFileSingleEn.write(line)
outFileSingleEn.close()

for line in outFileMultiCheStrings:
    outFileMultiChe.write(line)
outFileMultiChe.close()

for line in outFileMultiEnStrings:
    outFileMultiEn.write(line)
outFileMultiEn.close()





