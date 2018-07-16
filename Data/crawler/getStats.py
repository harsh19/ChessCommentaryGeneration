import pickle
import sys
import math
from collections import defaultdict
import textUtils

all_links=pickle.load(open("./saved_files/saved_links.p","r"))
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

for i,link in enumerate(all_links):
    total+=1
    
    pageLength=0
    if extra_links[i]>0:
        pageLength=1+(extra_links[i]-1)
    else:
        pageLength=1

    if pageLength>1:
        moreThanOneLong+=1
    
    totalPages+=pageLength

    for pageNo in range(pageLength):
        if pageNo==0:
            pageObjName="./outputs/saved"+str(i)+".obj"
        else:
            pageObjName="./outputs/saved"+str(i)+"_"+str(pageNo)+".obj"
        try:
            #This is one page
            data=pickle.load(open(pageObjName,"r"))
            totalMoves+=len(data)
            
            for elem in data:
                moves=elem[0]
                board=elem[1]
                comment=elem[2].encode('ascii','replace')
                commentWords=comment.split()
                
                if len(commentWords)>5:
                    longMoves+=1
                totalTokens+=len(commentWords)
                for word in commentWords:
                    word_frequencies[word]+=1


        except:
            break

    #break


wids["<UNK>"]=0
unkFrequency=0.0
rareWords=[]

for word,freq in word_frequencies.items():
    if freq>freqThreshold:
        wids[word]=len(wids)
    else:
        unkFrequency+=freq
        rareWords.append(word)

for word in rareWords:
    del word_frequencies[word]

word_frequencies["<UNK>"]=unkFrequency

entropy=0.0
for word,freq in word_frequencies.items():
    p_i=(freq+0.0)/(totalTokens)
    entropy+=p_i*math.log(p_i)


averagePages=totalPages/total
averageMoves=totalMoves/total

print "Total Games",total
print "More than One Long Games",moreThanOneLong
print "Total Pages",totalPages
print "Average Pages",averagePages
print "Total Moves",totalMoves
print "Long Moves",longMoves
print "Average Moves",averageMoves
print "Word Types",len(wids)
print "Rare Words",len(rareWords)
print "Word Tokens",totalTokens
print "Average Comment Length",(totalTokens+0.0)/(totalMoves+0.0)
print "Entropy of Freq",entropy
