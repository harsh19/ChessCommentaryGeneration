import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict


from utils import utilities as torch_utils
import numpy as np
import random
import math
import datetime
import gc
import sys

from models.SeqToSeqAttn import SeqToSeqAttn
from models.MultiSeqToSeq import MultiSeqToSeqAttn
from models.CNNtoSeq import CNNSeqToSeqAttn

class Solver:

    def __init__(self,cnfg):
        torch.manual_seed(1)
        random.seed(7867567)
        if cnfg.problem=="MT":
            from utils import readData as readData
            cnfg.srcLang="de"
            cnfg.tgtLang="en"
            cnfg.taskInfo="en-de.low"
        elif cnfg.problem=="CHESS":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.single"

        cnfg.readData=readData
        self.cnfg=cnfg

        #Create Language Objects
        self.cnfg.srcLangObj=self.cnfg.readData.Lang(self.cnfg.srcLang,self.cnfg.min_src_frequency,taskInfo=cnfg.taskInfo)
        self.cnfg.tgtLangObj=self.cnfg.readData.Lang(self.cnfg.tgtLang,self.cnfg.min_tgt_frequency,taskInfo=cnfg.taskInfo)
        self.cnfg.srcLangObj.initVocab("train")
        self.cnfg.tgtLangObj.initVocab("train")

    def splitData(self, data, batch_size=-1):

        if batch_size==-1:
            batch_size = self.cnfg.batch_size

        data_inputs=data[0]
        m = len(data_inputs[0])
        data_targets=data[1]
        cnfg=self.cnfg

        all_data = []
        all_masks=[]
        for j in range(m):
            src,src_masks=torch_utils.splitBatches(train=[data_input[j] for data_input in data_inputs], batch_size=batch_size,padSymbol=cnfg.garbage,method="pre")
            all_data.append(src)
            all_masks.append(src_masks)
        tgt_batches,tgt_masks=torch_utils.splitBatches(train=data_targets,batch_size=batch_size,padSymbol=cnfg.garbage,method="post")

        return [all_data, all_masks], [tgt_batches,tgt_masks]


    def main(self, typ="two_tuple"):
        self.typ=typ

        #Saving object variables as locals for quicker access
        cnfg=self.cnfg
        modelName=self.cnfg.modelName
        readData=self.cnfg.readData
        srcLangObj=self.cnfg.srcLangObj
        tgtLangObj=self.cnfg.tgtLangObj
        wids_src=srcLangObj.wids
        wids_tgt=tgtLangObj.wids
        print "src vocab size:",len(wids_src)
        print "tgt vocab size:",len(wids_tgt)

        # Define model. All modes.
        if cnfg.cudnnBenchmark:
            torch.backends.cudnn.benchmark=True
        print "Declaring Model, Loss, Optimizer"
        #model=SeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt)
        if typ=="two_tuple":
            model=MultiSeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt,encoder_configurations=['birnn','birnn'],typ=typ)
        elif typ=="entire_as_tuple":
            model=MultiSeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt,encoder_configurations=['birnn'],typ=typ)
        elif typ=="three_tuple":
            model=CNNSeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt,encoder_configurations=['cnn','cnn','birnn'],typ=typ)
            #model=MultiSeqToSeqAttn(cnfg,wids_src=wids_src,wids_tgt=wids_tgt,encoder_configurations=['birnn','birnn','birnn'],typ=typ)
        else:
            print "---------not supported typ "
            return
        loss_function=nn.NLLLoss(ignore_index=1,size_average=False)
        if torch.cuda.is_available():
            print "CUDA AVAILABLE. Making adjustments"
            model.cuda()
            loss_function=loss_function.cuda()
        print "DEFINING OPTIMIZER"
        optimizer=None
        if cnfg.optimizer_type=="SGD":
            optimizer=optim.SGD(model.getParams(),lr=0.05)
        elif cnfg.optimizer_type=="ADAM":
            optimizer=optim.Adam(model.getParams())

        ### train and validation data
        if cnfg.mode=="train" or cnfg.mode=="trial":

            train_src=srcLangObj.read_corpus("train", typ=typ)
            train_tgt=tgtLangObj.read_corpus("train", typ="sequence")
            valid_src=srcLangObj.read_corpus(mode="valid", typ=typ)
            valid_tgt=tgtLangObj.read_corpus(mode="valid", typ="sequence")
            if cnfg.mode=="train":
                train_src,train_tgt=train_src[:cnfg.max_train_sentences],train_tgt[:cnfg.max_train_sentences]
            print "training size:",len(train_src)
            print "valid size:",len(valid_src)
            train=zip(train_src,train_tgt) #zip(train_src,train_tgt)
            valid=zip(valid_src,valid_tgt) #zip(train_src,train_tgt)

            train.sort(key=lambda x:len(x[0][0])) # sort by target length
            valid.sort(key=lambda x:len(x[0][0])) # sort by target length
            train_src,train_tgt=[x[0] for x in train],[x[1] for x in train]
            valid_src,valid_tgt=[x[0] for x in valid],[x[1] for x in valid]

            print "-------------- TUPLE"

            src, trgt = self.splitData(data=[train_src, train_tgt])
            train_tgt_batches,train_tgt_masks = trgt
            train_src_batches, train_src_masks = src

            src, trgt = self.splitData(data=[valid_src, valid_tgt])
            valid_src_batches, valid_src_masks = src
            valid_tgt_batches,valid_tgt_masks = trgt

            #Dump useless references
            train=None
            valid=None
            #Sanity check
            assert (len(train_tgt_batches)==len(train_src_batches[0]))
            assert (len(valid_tgt_batches)==len(valid_src_batches[0]))
            print "Training Batches:",len(train_tgt_batches)
            print "Validation Batches:",len(valid_tgt_batches)

            if cnfg.mode=="trial":

                print "Running Sample Batch"
                print "Target Batch Shape:",train_tgt_batches[30].shape
                print "Target Mask Shape:",train_tgt_masks[30].shape
                sample_src_batch=[c[30] for c in train_src_batches]
                sample_tgt_batch=train_tgt_batches[30]
                sample_tgt_mask=train_tgt_masks[30]
                sample_src_mask=[c[30] for c in train_src_masks]
                print datetime.datetime.now()
                model.zero_grad()
                #loss=model.forward(sample_src_batch,sample_tgt_batch,sample_src_mask,sample_tgt_mask,loss_function)
                loss=model(sample_src_batch,sample_tgt_batch,sample_src_mask,sample_tgt_mask,loss_function)
                #loss=model.forward(sample_src_batch,sample_tgt_batch,sample_src_mask,sample_mask,loss_function)
                print loss
                loss.backward()
                optimizer.step()
                print datetime.datetime.now()
                print "Done Running Sample Batch"

            elif cnfg.mode=="train":

                print "====TRAIN MODE==="
                print "Start Time:",datetime.datetime.now()

                for epochId in range(cnfg.NUM_EPOCHS):

                    batch_indices = range(len(train_tgt_batches))
                    random.shuffle( batch_indices )
                    for j,batchId in enumerate(batch_indices):
                        src_batch = [ele[batchId] for ele in train_src_batches]
                        src_mask = [ele[batchId] for ele in train_src_masks]
                        tgt_batch, tgt_mask= train_tgt_batches[batchId], train_tgt_masks[batchId]
                        batchLength=src_batch[0].shape[1] # src_batch[0] corresponds to prevboard
                        batchSize=src_batch[0].shape[0]
                        tgtBatchLength=tgt_batch.shape[1]

                        if batchLength<cnfg.MAX_SEQ_LEN and batchSize>1 and tgtBatchLength<cnfg.MAX_TGT_SEQ_LEN:
                            model.zero_grad()
                            loss=model(src_batch,tgt_batch,src_mask,tgt_mask,loss_function)
                            #loss=model.forward(src_batch,tgt_batch,src_mask,tgt_mask,loss_function)
                            if cnfg.mem_optimize:
                                del src_batch,tgt_batch,src_mask,tgt_mask
                            loss.backward()
                            if cnfg.mem_optimize:
                                del loss
                            optimizer.step()

                        if j%cnfg.PRINT_STEP==0:
                            print "Batch No:",j," Time:",datetime.datetime.now()
                            # validation perplexity
                            totalValidationLoss=0.0
                            NUM_TOKENS=0.0
                            m = len(valid_tgt_batches)
                            for batchId_val in range(m):
                                src_batch = [ele[batchId_val] for ele in valid_src_batches]
                                src_mask = [ele[batchId_val] for ele in valid_src_masks]
                                tgt_batch, tgt_mask= valid_tgt_batches[batchId_val], valid_tgt_masks[batchId_val]

                                model.zero_grad()
                                loss=model.forward(src_batch,tgt_batch,src_mask,tgt_mask,loss_function,inference=True)
                                if cnfg.normalizeLoss:
                                    totalValidationLoss+=(loss.data.cpu().numpy())*np.sum(tgt_mask)
                                else:
                                    totalValidationLoss+=(loss.data.cpu().numpy())
                                NUM_TOKENS+=np.sum(tgt_mask)
                                if cnfg.mem_optimize:
                                    del src_batch,tgt_batch,src_mask,tgt_mask,loss
                            perplexity=math.exp(totalValidationLoss/NUM_TOKENS)
                            print "Epoch:",epochId," Total Validation Loss:",totalValidationLoss," Perplexity:",perplexity

                    # save model after epoch end
                    model.save_checkpoint(modelName+"_"+str(epochId),optimizer)

                print "End Time:",datetime.datetime.now()

        elif cnfg.mode=="inference":

            test_src=srcLangObj.read_corpus(mode="test", typ=typ)
            test_tgt=tgtLangObj.read_corpus(mode="test", typ="sequence")
            print "--------------THREE TUPLE"

            src, trgt = self.splitData(data=[test_src, test_tgt], batch_size=1)
            test_tgt_batches, test_tgt_masks = trgt
            test_src_batches, test_src_masks = src

            print len(test_tgt_batches)
            print len(test_src_batches[0])
            assert (len(test_tgt_batches)==len(test_src_batches[0]))
            print "Test Points:",len(test_tgt_batches)

            model.load_from_checkpoint(modelName)
            #Evaluate on test first
            model.decodeAll(test_src_batches,test_src_masks,modelName,method="greedy",evalMethod="BLEU",suffix="test")
            #Also on valid
