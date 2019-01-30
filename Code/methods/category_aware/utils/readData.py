import sys
from collections import defaultdict
import re

class Lang():
    def __init__(self,name,min_frequency,dataPath="data/",taskInfo="en-de.low"):
        self.name=name
        self.min_frequency=min_frequency
        self.wids=defaultdict(lambda : len(self.wids)) #word2index
        self.dataPath=dataPath
        self.taskInfo=taskInfo

    def initVocab(self,mode):
        fileName=self.dataPath+".".join([mode,self.taskInfo,self.name])

        self.unk_word="<unk>"
        self.wids["<s>"]=0
        self.wids[self.unk_word]=1
        self.wids["</s>"]=2
        self.wids["<GARBAGE>"]=3

        word_frequencies=defaultdict(int)
        for line in open(fileName):
            words=line.split()
            for word in words:
                word_frequencies[word]+=1

        for word,freq in word_frequencies.items():
            if freq>self.min_frequency:
                self.wids[word]=len(self.wids)

    def read_corpus(self,mode,typ="sequence"): # type {sequence, three_tuple}
        file_name=self.dataPath+".".join([mode,self.taskInfo,self.name])
        sentences=[]

        if typ=="sequence":
            for line in open(file_name):
                words=line.split()
                #words.insert(0,"<s>")
                words.append("</s>")
                sentence=[self.wids[word] if word in self.wids else 1 for word in words]
                sentences.append(sentence)
            return sentences
 
        elif typ=="sequence_category":
            for line in open(file_name):
                words=line.split()
                #words.insert(0,"<s>")
                words.append("</s>")
                sentence=[self.wids[word] if word in self.wids else 1 for word in words]
                sentences.append(sentence)
            category_fname = self.dataPath+".".join([mode,self.taskInfo]) + ".categories"
            category_data = open(category_fname,"r").readlines()
            category_data = [ s.strip() for s in category_data ]
            assert len(category_data) == len(sentences)
            data = zip(sentences, category_data)
            return data

        elif typ=="three_tuple":
            prev_board=[]
            cur_board=[]
            moves=[]
            for line in open(file_name):
                components=re.split(' <EOC> | <EOP> ', line)
                assert len(components)==3 # cur, prev, move
                tmp = []
                #words.append("</s>")
                for component in components:
                    words = component.split()
                    sentence=[self.wids[word] if word in self.wids else self.wids[self.unk_word] for word in words]
                    tmp.append(sentence)
                prev_board.append(tmp[0])
                cur_board.append(tmp[1])
                moves.append(tmp[2])
            data = zip(prev_board, cur_board, moves)
            return data

        elif typ=="two_tuple":
            prev_board=[]
            cur_board=[]
            moves=[]
            for line in open(file_name):
                components=re.split(' <EOC> | <EOP> ', line)
                assert len(components)==3 # cur, prev, move
                tmp = []
                #words.append("</s>")
                for component in components:
                    words = component.split()
                    sentence=[self.wids[word] if word in self.wids else self.wids[self.unk_word] for word in words]
                    tmp.append(sentence)
                prev_board.append(tmp[0])
                cur_board.append(tmp[1])
                moves.append(tmp[2])
            data = zip(prev_board, cur_board)
            return data

        elif typ=="entire_as_tuple":
            for line in open(file_name):
                words=line.split()
                #words.insert(0,"<s>")
                words.append("</s>")
                sentence=[self.wids[word] if word in self.wids else 1 for word in words]
                sentences.append([sentence])
            return sentences



def read_corpus(wids,mode="train",update_dict=True,min_frequency=3,language="en"):
    fileName=None
    if mode=="train":
        fileName="train.en-de.low."+language
    elif mode=="valid":
        fileName="valid.en-de.low."+language
    elif mode=="test":
        fileName="test.en-de.low."+language
    else:
        fileName="blind.en-de.low."+language
    fileName="data/"+fileName


    if update_dict:
        wids["<s>"]=0
        wids["<unk>"]=1
        wids["</s>"]=2
        wids["<GARBAGE>"]=3

        word_frequencies=defaultdict(int)
        for line in open(fileName):
            words=line.split()
            for word in words:
                word_frequencies[word]+=1

        for word,freq in word_frequencies.items():
            if freq>min_frequency:
                wids[word]=len(wids)

    sentences=[]

    for line in open(fileName):
        words=line.split()
        #words.insert(0,"<s>")
        words.append("</s>")
        sentence=[wids[word] if word in wids else 1 for word in words]
        sentences.append(sentence)

    return sentences




if __name__=="__main__":
    wids=defaultdict(lambda: len(wids))
    train_sentences=read_corpus(wids,mode="train",update_dict=True)
    valid_sentences=read_corpus(wids,mode="valid",update_dict=False)
    test_sentences=read_corpus(wids,mode="test",update_dict=False)
