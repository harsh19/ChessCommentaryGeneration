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

from models.LM import LM
import pickle

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
        elif cnfg.problem=="CHESS0":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.0"
        elif cnfg.problem=="CHESS0SIMPLE":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.0simple"
        elif cnfg.problem=="CHESS0ATTACK":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.0attack"
        elif cnfg.problem=="CHESS0ATTACKENTITIES":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en.entities"
            cnfg.taskInfo="che-eng.0attack"
        elif cnfg.problem=="CHESS0SIMPLEENTITIES":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en.entities"
            cnfg.taskInfo="che-eng.0simple"
        elif cnfg.problem=="CHESS1SIMPLE":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.1simple"
        elif cnfg.problem=="CHESS1ATTACK":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.1attack"
        elif cnfg.problem=="CHESS2SIMPLE":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.2simple"
        elif cnfg.problem=="CHESS2ATTACK":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.2attack"



        elif cnfg.problem=="CHESSLABEL":
            from utils import readData as readData
            cnfg.srcLang="che"
            cnfg.tgtLang="en"
            cnfg.taskInfo="che-eng.aspect"


      
        cnfg.readData=readData
        self.cnfg=cnfg

        #Create Language Objects
        #self.cnfg.srcLangObj=self.cnfg.readData.Lang(self.cnfg.srcLang,self.cnfg.min_src_frequency)
        #self.cnfg.tgtLangObj=self.cnfg.readData.Lang(self.cnfg.tgtLang,self.cnfg.min_tgt_frequency)
        self.cnfg.srcLangObj=self.cnfg.readData.Lang(self.cnfg.srcLang,self.cnfg.min_src_frequency,taskInfo=cnfg.taskInfo)
        self.cnfg.tgtLangObj=self.cnfg.readData.Lang(self.cnfg.tgtLang,self.cnfg.min_tgt_frequency,taskInfo=cnfg.taskInfo)
        self.cnfg.srcLangObj.initVocab("train")
        self.cnfg.tgtLangObj.initVocab("train")

    
    def score(self,idList):
        test_tgt_batches,test_tgt_masks=torch_utils.splitBatches(train=[idList,],batch_size=1,padSymbol=self.cnfg.garbage,method="post")
        loss=self.model.forward(test_tgt_batches[0],test_tgt_masks[0],self.loss_function,inference=True)
        return loss.data.cpu().numpy()[0]

    def main(self):
        
        #Saving object variables as locals for quicker access
        cnfg=self.cnfg
        modelName=self.cnfg.modelName
        readData=self.cnfg.readData
        srcLangObj=self.cnfg.srcLangObj
        tgtLangObj=self.cnfg.tgtLangObj
        wids_src=srcLangObj.wids
        wids_tgt=tgtLangObj.wids

        
        train_src=srcLangObj.read_corpus("train")
        train_tgt=tgtLangObj.read_corpus("train")

        if cnfg.mode!="inference":
            valid_src=srcLangObj.read_corpus(mode="valid")
            valid_tgt=tgtLangObj.read_corpus(mode="valid")

        test_src=srcLangObj.read_corpus(mode="test")
        test_tgt=tgtLangObj.read_corpus(mode="test")



        train_src,train_tgt=train_src[:cnfg.max_train_sentences],train_tgt[:cnfg.max_train_sentences]
        print "src vocab size:",len(wids_src)
        print "tgt vocab size:",len(wids_tgt)
        print "training size:",len(train_src)
        if cnfg.mode!="inference":
            print "valid size:",len(valid_src)

        train=zip(train_src,train_tgt) #zip(train_src,train_tgt)
        if cnfg.mode!="inference":
            valid=zip(valid_src,valid_tgt) #zip(train_src,train_tgt)
        

        train.sort(key=lambda x:len(x[0]))
        
        if cnfg.mode!="inference":
            valid.sort(key=lambda x:len(x[0]))


        train_src,train_tgt=[x[0] for x in train],[x[1] for x in train]
        
        if cnfg.mode!="inference":
            valid_src,valid_tgt=[x[0] for x in valid],[x[1] for x in valid]
        


        train_src_batches,train_src_masks=torch_utils.splitBatches(train=train_src,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="pre")
        train_tgt_batches,train_tgt_masks=torch_utils.splitBatches(train=train_tgt,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="post")
        if cnfg.mode!="inference":
            valid_src_batches,valid_src_masks=torch_utils.splitBatches(train=valid_src,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="pre")
            valid_tgt_batches,valid_tgt_masks=torch_utils.splitBatches(train=valid_tgt,batch_size=cnfg.batch_size,padSymbol=cnfg.garbage,method="post")
        
        test_src_batches,test_src_masks=torch_utils.splitBatches(train=test_src,batch_size=1,padSymbol=cnfg.garbage,method="pre")
        test_tgt_batches,test_tgt_masks=torch_utils.splitBatches(train=test_tgt,batch_size=1,padSymbol=cnfg.garbage,method="post")


        #Dump useless references
        train=None
        valid=None
        #Sanity check
        assert (len(train_tgt_batches)==len(train_src_batches))
        if cnfg.mode!="inference":
            assert (len(valid_tgt_batches)==len(valid_src_batches))
        assert (len(test_tgt_batches)==len(test_src_batches))

        print "Training Batches:",len(train_tgt_batches)
        if cnfg.mode!="inference":
            print "Validation Batches:",len(valid_tgt_batches)
        print "Test Points:",len(test_src_batches)

        if cnfg.cudnnBenchmark:
            torch.backends.cudnn.benchmark=True
        #Declare model object
        print "Declaring Model, Loss, Optimizer"
        model=LM(cnfg,wids_src=wids_src,wids_tgt=wids_tgt)
        loss_function=nn.NLLLoss(ignore_index=1,size_average=False)
        self.loss_function=loss_function
        if torch.cuda.is_available():
            model.cuda()
            loss_function=loss_function.cuda()
        optimizer=None
        if cnfg.optimizer_type=="SGD":
            optimizer=optim.SGD(model.getParams(),lr=0.05)
        elif cnfg.optimizer_type=="ADAM":
            optimizer=optim.Adam(model.getParams())

        if cnfg.mode=="trial":
            print "Running Sample Batch" 
            print "Source Batch Shape:",train_src_batches[30].shape
            print "Source Mask Shape:",train_src_masks[30].shape
            print "Target Batch Shape:",train_tgt_batches[30].shape
            print "Target Mask Shape:",train_tgt_masks[30].shape
            sample_src_batch=train_src_batches[30]
            sample_tgt_batch=train_tgt_batches[30]
            sample_mask=train_tgt_masks[30]
            sample_src_mask=train_src_masks[30]
            print datetime.datetime.now() 
            model.zero_grad()
            loss=model.forward(sample_src_batch,sample_tgt_batch,sample_src_mask,sample_mask,loss_function)
            print loss
            loss.backward()
            optimizer.step()
            print datetime.datetime.now()
            print "Done Running Sample Batch"

        train_batches=zip(train_src_batches,train_tgt_batches,train_src_masks,train_tgt_masks)
        if cnfg.mode!="inference":
            valid_batches=zip(valid_src_batches,valid_tgt_batches,valid_src_masks,valid_tgt_masks)

        train_src_batches,train_tgt_batches,train_src_masks,train_tgt_masks=None,None,None,None
        if cnfg.mode!="inference":
            valid_src_batches,valid_tgt_batches,valid_src_masks,valid_tgt_masks=None,None,None,None
        
        if cnfg.mode=="train" or cnfg.mode=="LM":
            print "Start Time:",datetime.datetime.now()     
            for epochId in range(cnfg.NUM_EPOCHS):
                random.shuffle(train_batches)
                for batchId,batch in enumerate(train_batches):
                    src_batch,tgt_batch,src_mask,tgt_mask=batch[0],batch[1],batch[2],batch[3]
                    batchLength=src_batch.shape[1]
                    batchSize=src_batch.shape[0]
                    tgtBatchLength=tgt_batch.shape[1]
                    if batchLength<cnfg.MAX_SEQ_LEN and batchSize>1 and tgtBatchLength<cnfg.MAX_TGT_SEQ_LEN:
                        model.zero_grad()
                        loss=model.forward(tgt_batch,tgt_mask,loss_function)
                        if cnfg.mem_optimize:
                            del src_batch,tgt_batch,src_mask,tgt_mask
                        loss.backward()
                        if cnfg.mem_optimize:
                            del loss
                        optimizer.step()               
                    if batchId%cnfg.PRINT_STEP==0:
                        print "Batch No:",batchId," Time:",datetime.datetime.now()

                totalValidationLoss=0.0
                NUM_TOKENS=0.0
                for batchId,batch in enumerate(valid_batches):
                    src_batch,tgt_batch,src_mask,tgt_mask=batch[0],batch[1],batch[2],batch[3]
                    model.zero_grad()
                    loss=model.forward(tgt_batch,tgt_mask,loss_function,inference=True)
                    if cnfg.normalizeLoss:
                        totalValidationLoss+=(loss.data.cpu().numpy())*np.sum(tgt_mask)
                    else:
                        totalValidationLoss+=(loss.data.cpu().numpy())
                    NUM_TOKENS+=np.sum(tgt_mask)
                    if cnfg.mem_optimize:
                        del src_batch,tgt_batch,src_mask,tgt_mask,loss
                
                model.save_checkpoint(modelName+"_"+str(epochId),optimizer)

                perplexity=math.exp(totalValidationLoss/NUM_TOKENS)
                print "Epoch:",epochId," Total Validation Loss:",totalValidationLoss," Perplexity:",perplexity
            print "End Time:",datetime.datetime.now()

        elif cnfg.mode=="inference":
            model.load_from_checkpoint(modelName)
            #Evaluate on test first
            model.decodeAll(test_src_batches,modelName,method=cnfg.method,evalMethod="BLEU",suffix="test")
            #Also on valid
            valid_src=srcLangObj.read_corpus(mode="valid")
            valid_src_batches,valid_src_masks=torch_utils.splitBatches(train=valid_src,batch_size=1,padSymbol=cnfg.garbage,method="pre")
            model.decodeAll(valid_src_batches,modelName,method=cnfg.method,evalMethod="BLEU",suffix="valid")

        elif cnfg.mode=="saveLM":
            "Checking out LM"
            self.model=model
            model.load_from_checkpoint(modelName)
            loss=model.forward(test_tgt_batches[0],test_tgt_masks[0],loss_function,inference=True)
            print loss.data.cpu().numpy()
            print self.score(test_tgt[0])
            print "Checked out LM"
            return self
