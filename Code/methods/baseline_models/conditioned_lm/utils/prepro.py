# logging
import logging
logging.basicConfig(level=logging.DEBUG)

# general python
import numpy as np
import scipy.misc
import pickle
import random
import pickle
import math
import json

# pytorch
import torch

####################################################################

class Lang():
	def __init__(self, name, min_frequency):
		self.name = name
		self.min_frequency = min_frequency
		self.pad_word = "<pad>"
		self.start = "<start>"
		self.end = "<end>"
		self.unk = "<unk>"
		self.word_frequencies = {}

	def getStartTokenIdx(self):
		return self.word2idx[self.start]

	def getPadTokenIdx(self):
		return self.word2idx[self.pad_word]

	def getEndTokenIdx(self):
		return self.word2idx[self.end]

	def _addSentence(self, sentence):
		words=sentence.split() #TODO
		for word in words:
			if word not in self.word_frequencies:
				self.word_frequencies[word]= 1
			else:
				self.word_frequencies[word]+= 1

	def getSentenceToIndexList(self, sentence):
		ret = []
		words = sentence.strip().split() #TODO
		for word in words:
			if word in self.word2idx:
				ret.append(self.word2idx[word])
			else:
				#print "** ", word
				ret.append(self.word2idx[self.unk])
		return ret

	def getSentenceFromIndexList(self, lst_of_word_idx):
		ret=  ""
		for word_idx in lst_of_word_idx:
			if word_idx==self.getEndTokenIdx() or word_idx==self.getPadTokenIdx():
				break
			ret = ret + self.idx2word[word_idx] + " "
		return ret.strip()

	def setupVocab(self, data):

		logging.info("="*33)
		logging.info("Lang: setupVocab()")

		# calculate frequencies
		for sentence in data:
			self._addSentence(sentence)

		# init
		self.word2idx = {self.start:1,self.pad_word:0,self.end:2,self.unk:3}
		self.idx2word = {}
		self.vocab_cnt = 4

		# setup vocab
		for word,freq in self.word_frequencies.items():
			if freq>=self.min_frequency:
				self.word2idx[word]= self.vocab_cnt
				self.vocab_cnt += 1
			else:
				pass
				#print " *** ", word, freq, self.min_frequency
		self.idx2word = {idx:w for w,idx in self.word2idx.items()}

		logging.info("Vocab size = " + str(self.vocab_cnt) )
		#print "word2idx = ", self.word2idx



####################################################################

class Prepro:

	def __init__(self, params, lang="en"):
		self.lang = Lang(lang, params.min_tgt_frequency)
		self.params = params

	def _process(self, data):
		return [s.lower() for s in data]

	def _loadData(self, src):
		data = open(src,"r").readlines()
		data = [row.strip() for row in data]
		data = self._process(data)
		return data

	def _combine(self, sentences, features, debug):
		#ret = [ [xx,None] for xx,yy in zip(x,y) ]
		if debug:
			sentences = sentences[:90]
		#ret = [ [sentence, np.ones(11)] for sentence in sentences ]
		ret = [ [sentence, feats] for sentence,feats in zip(sentences,features) ]
		return ret

	def _loadFeats(self, dir, data_typ, feat_typ, split):
		logging.debug("Loading from " + dir+split+"."+data_typ+"."+feat_typ + ".all_feats.pickle")
		return pickle.load( open(dir+split+"."+data_typ+"."+feat_typ + ".all_feats.pickle","r") )

	def _getFeats(self, split):
		feats_dir = self.params.feats_dir
		feats_mask = [int(s) for s in self.params.feats]
		feat_types = ["move","score","threat"]
		ret = []
		for i,mask in enumerate(feats_mask):
			if mask==1:
				feats = self._loadFeats(feats_dir, self.params.typ, feat_types[i], split)
				np.nan_to_num(feats, copy=False)
				feats = np.array( feats )
				#if i==0: # move
				#	feats = feats.reshape(-1,1)
				ret.append(feats)
				logging.debug("[_getFeats(): split:" + split + "; feat type: " +feat_types[i]+ " feats: " + str(feats.shape) )
		ret = np.hstack(ret)
		return ret


	def getData(self, src_dir = "../../../data/", typ="che-eng.0", lang="en"):
		
		splits = ["train","valid","test"]
		params = self.params

		train_feats = self._getFeats('train')
		self.params.encoder_feat_size = train_feats.shape[1]
		logging.info("Setting encoder_feat_size = " + str(self.params.encoder_feat_size) )
		val_feats = self._getFeats('valid')
		test_feats = self._getFeats('test')

		train_data = self._combine( self._loadData(src_dir + "train." + typ + "." + lang), train_feats, params.debug )
		val_data = self._combine( self._loadData(src_dir + "valid." + typ + "." + lang), val_feats, params.debug )
		test_data = self._combine( self._loadData(src_dir + "test." + typ + "." + lang), test_feats, params.debug )
		
		logging.info("[prepro.getData(): setupVocab")
		train_data_all = [row[0] for row in train_data] #self._loadData(src_dir + "train." + typ + "." + lang)
		self.lang.setupVocab(train_data_all)

		self.train_split = train_data 
		self.val_split = val_data 
		self.test_split = test_data

		logging.info("[prepro.getData()]: #train = " + str(len(self.train_split)) )
		logging.info("[prepro.getData()]: #val = " + str(len(self.val_split)) )
		logging.info("[prepro.getData()]: #test = " + str(len(self.test_split)) )
		self.data_lens = {'train':len(self.train_split), 'val':len(self.val_split), 'test':len(self.test_split)}

		logging.info("[prepro.getData()]: train data sample = " + str(self.train_split[23][0]) ) 
		logging.info("[prepro.getData()]: train data sample = " + str( self._getSentenceRepresentation(self.train_split[23][0]) ) ) 


	def getLang(self):
		return self.lang

	def getNumberOfBatches(self, split, batch_size):
		return ( self.data_lens[split] + batch_size - 1 ) / batch_size

	def _getSentenceRepresentation(self, sentence):
		sentence = sentence + " " + self.lang.end 
		# since start is anyway added while training/decoding from lstm
		rep = self.lang.getSentenceToIndexList(sentence)
		return rep

	def shuffle_train(self):
		indices = np.arange(len(self.train_split))
		np.random.shuffle(indices)
		self.train_split= [ self.train_split[idx] for idx in indices ]

	def getBatch(self, split, batch_size, i):

		if split not in ["train","val","test"]:
			print "INVALID 'split'"
			return
		if split=="train": split_vals = self.train_split[i*batch_size:(i+1)*batch_size]
		if split=="val": split_vals = self.val_split[i*batch_size:(i+1)*batch_size]
		if split=="test": split_vals = self.test_split[i*batch_size:(i+1)*batch_size]

		train_x = np.array( [ vals[1] for vals in split_vals ] )
		train_y =  [ self._getSentenceRepresentation(vals[0]) for vals in split_vals ]
		maxlen = max([len(train_y_i) for train_y_i in train_y])
		pad_token = self.lang.getPadTokenIdx()
		train_and_mask_y = [ self._padSeq(train_y_i, maxlen, pad_token, method="post") for train_y_i in train_y ]
		train_y = [ train_and_mask_y_i[0] for train_and_mask_y_i in train_and_mask_y ]
		mask_y = [ train_and_mask_y_i[1] for train_and_mask_y_i in train_and_mask_y ]
		train_y = np.array( train_y ) #TODO:
		return train_x, train_y, mask_y # currently not using mask as loss function is already set to ignore pad index


	def _maskSeq(self, seq, desired_length, pad_symbol, method="pre"):
		seq_length=len(seq)
		mask=[1,]*desired_length
		if len(seq)<desired_length:
			if method=="post":
				mask=[1,]*seq_length+[0,]*(desired_length-seq_length)
			else:
				mask=[0,]*(desired_length-seq_length)+[1,]*seq_length
		return mask

	def _padSeq(self, seq, desired_length, pad_symbol, method="pre"):
		seq_length=len(seq)
		mask=[1,]*desired_length
		if len(seq)<desired_length:
			if method=="post":
				seq=seq+[pad_symbol,]*(desired_length-seq_length)
				mask=[1,]*seq_length+[0,]*(desired_length-seq_length)
			else:
				seq=[pad_symbol,]*(desired_length-seq_length)+seq
				mask=[0,]*(desired_length-seq_length)+[1,]*seq_length
		return seq, mask
