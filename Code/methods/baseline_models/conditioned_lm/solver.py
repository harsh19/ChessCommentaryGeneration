# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.DEBUG)

import pickle
import time
import numpy
import os
from collections import defaultdict

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

import numpy as np
import random
import math
import datetime
import gc
import sys


method_type=["cond_lm", "lm"][0]
import clm
from utils import prepro


class Solver:

    def __init__(self, params):
        torch.manual_seed(1)
        random.seed(7867567)
        self.params = params

        self.prepro = prepro.Prepro(params, lang=params.lang)
        #self.prepro.getData(src_dir = "../../data/", typ="che-eng.0", lang="en")
        self.prepro.getData(src_dir = params.data_dir, typ=params.typ, lang=params.lang)
        params.vocab_size = len(self.prepro.lang.word2idx)

        #---- create a simple model
        start_symbol_idx = self.prepro.lang.getStartTokenIdx()
        end_symbol_idx = self.prepro.lang.getEndTokenIdx()
        self.model = clm.CLM(params, start_symbol_idx, end_symbol_idx)
        self.loss_function = nn.NLLLoss(ignore_index=0,size_average=False )
        if torch.cuda.is_available():
            logging.info("CUDA AVAILABLE. Making adjustments")
            self.model.cuda()
            self.loss_function = self.loss_function.cuda()

        # optimizer
        logging.info( "DEFINING OPTIMIZER" )
        self.optimizer=None
        if params.optimizer_type=="SGD":
            self.optimizer = optim.SGD(self.model.parameters(),lr=0.05)
        elif params.optimizer_type=="ADAM":
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            print "unsupported choice of optimizer"


    def _trainBatch(self, batch_x, batch_y):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.model(batch_x, gt_output=batch_y, mode='train', loss_function=self.loss_function)
        #print "loss = ",loss
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def _getLoss(self, batch_x, batch_y):
        loss = self.model(batch_x, gt_output=batch_y, mode='train', loss_function=self.loss_function)
        return loss.data[0]

    def _decodeBatch(self, batch_x, batch_y=None, get_loss=False, decoding_type="greedy"):
        #print "batch_y = ",batch_y
        outputs = self.model(batch_x, gt_output=batch_y, mode='decode', loss_function=self.loss_function,\
         get_loss=get_loss, max_len_decode=self.params.max_decode_length, decoding_type=decoding_type)
        return outputs

    def train(self):
        logging.info("="*33)
        logging.info("Beginning training procedure")

        for epoch in range(self.params.num_epochs):

            logging.info("\n ------------- \n Epoch = "+str(epoch) + "-"*21 + "\n")
            epoch_loss = 0.0
            num_batches = self.prepro.getNumberOfBatches('train', self.params.batch_size)
            for batch_idx in range(num_batches):
                batch_x, batch_y, _ = self.prepro.getBatch(split='train',batch_size=self.params.batch_size,i=batch_idx)
                batch_x = torch.autograd.Variable( torch.FloatTensor(batch_x) )
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                #print "batch_x = ", batch_x, batch_y
                loss = self._trainBatch(batch_x, batch_y)
                epoch_loss+=loss

            logging.info("Epoch train loss = "+str(epoch_loss))
            val_loss = 0.0
            mask_y_sum = 0
            for i in range(self.prepro.getNumberOfBatches('val',self.params.batch_size)):
                batch_x, batch_y, mask_y = self.prepro.getBatch(split='val',batch_size=self.params.batch_size,i=i)
                batch_x = torch.autograd.Variable( torch.FloatTensor(batch_x) )
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                val_loss+= self._getLoss(batch_x, batch_y)
                mask_y_sum+= np.sum(mask_y)
            logging.info("Epoch val loss = "+str(val_loss))
            
            #print " mask_y_sum = ", mask_y_sum
            logging.info("Epoch val perplexity = "+str( np.exp(val_loss/mask_y_sum) ))
            val_bleu = self.getBleu(split='val', output_path = "./tmp/"+self.params.model_name+"_"+str(epoch))
            logging.info("After epoch "+str(epoch)+" : val bleu = " + str(val_bleu))

            self.saveModel(str(epoch))
            self.prepro.shuffle_train()

    def _decodeAll(self, split="val", decoding_type="greedy"): # 'val'
        all_outputs = []
        all_gt = []
        num_batches = self.prepro.getNumberOfBatches(split, self.params.batch_size)
        for batch_idx in range(num_batches):
            batch_x, batch_y, _ = self.prepro.getBatch(split=split, batch_size=self.params.batch_size, i=batch_idx)
            batch_x = torch.autograd.Variable( torch.FloatTensor(batch_x) )
            if torch.cuda.is_available():
                batch_x = batch_x.cuda()
            outputs = self._decodeBatch(batch_x, batch_y, decoding_type=decoding_type)
            all_outputs.extend(outputs)
            all_gt.extend(batch_y)
        return all_outputs, all_gt

    def _outputToFile(self, fpath, out_data):
        fw = open(fpath,"w")
        for row in out_data:
            row = row.strip()
            fw.write(row + "\n")
        fw.close()

    def getBleu(self, split="val", output_path="tmp/bleu"): # 'val'
        outputs, gt_outputs = self._decodeAll(split, decoding_type=self.params.decoding_type)
        #print "outputs = ",outputs
        text_outputs = [ self.prepro.lang.getSentenceFromIndexList(output) for output in outputs ]
        gt_text_outputs = [ self.prepro.lang.getSentenceFromIndexList(output) for output in gt_outputs ]
        #print "text outputs = ",text_outputs
        #print "text gt_outputs = ",gt_text_outputs
        fname_gt = output_path + "_" + split + ".gt"
        ##split_val = split
        ##if split=="val":
        ##    split_val = "valid"
        ##fname_gt = self.params.data_dir + split_val + "." + self.params.typ + "." + self.params.lang
        self._outputToFile( fname_gt, gt_text_outputs )
        fname_pred = output_path + "_" + split + ".pred"
        self._outputToFile(fname_pred, text_outputs )
        bleu = os.popen("perl multi-bleu.perl -lc "+ fname_gt +" < "+ fname_pred ).read()
        return bleu

    def saveModel(self, extra):
        checkpoint = self.model.state_dict()
        #checkpoint['optimizer'] = self.optimizer.state_dict()
        torch.save(checkpoint, "./tmp/"+self.params.model_name+"_"+extra+".ckpt")
        print "Saved Model"

    def loadSavedModel(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict({k:v for k,v in checkpoint.items() if k!="optimizer"})
        #if optimizer!=None:
        #    optimizer.load_state_dict(checkpoint['optimizer'])
        print "Loaded Model"


    def main(self):
        if self.params.mode=="train":
            self.train()
        elif self.params.mode=="decode":
            self.loadSavedModel(self.params.model_name)
            bleu = self.getBleu(split="test", output_path=self.params.model_name+".decode.")
            print "TEST BLEU = ", bleu
