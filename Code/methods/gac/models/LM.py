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

from modules import * 
from embeddingUtils import *

class LM():
    def __init__(self,cnfg,wids_src=None,wids_tgt=None):

        self.wids_src=wids_src
        self.wids_tgt=wids_tgt
        self.reverse_wids_src=torch_utils.reverseDict(wids_src)
        self.reverse_wids_tgt=torch_utils.reverseDict(wids_tgt)
        self.cnfg=cnfg
        self.cnfg.srcVocabSize=len(self.wids_src)
        self.cnfg.tgtVocabSize=len(self.wids_tgt)
        
        #self.encoder=EncoderRNN(self.wids_src,self.cnfg.srcVocabSize,self.cnfg.emb_size,self.cnfg.hidden_size,self.cnfg.use_LSTM,False)
        #if self.cnfg.use_reverse:
        #    self.revcoder=EncoderRNN(self.wids_src,self.cnfg.srcVocabSize,self.cnfg.emb_size,self.cnfg.hidden_size,self.cnfg.use_LSTM,True,reference_embeddings=self.encoder.embeddings)

        self.decoder=DecoderRNN(self.wids_tgt,cnfg.tgtVocabSize,cnfg.emb_size,cnfg.hidden_size)

        #if self.cnfg.initGlove:
        #    embed=self.decoder.embeddings.weight.data.cpu().numpy()
        #    embed=loadEmbedsAsNumpyObj("./glove/wiki.simple.vec",self.wids_tgt,embed)
        #    self.decoder.embeddings.weight.data.copy_(torch.from_numpy(embed))

        #if self.cnfg.use_attention and self.cnfg.use_downstream:
        #    self.W=LinearLayer(2*self.cnfg.hidden_size,self.cnfg.tgtVocabSize)
        #else:
        self.W=LinearLayer(self.cnfg.hidden_size,self.cnfg.tgtVocabSize)

    def zero_grad(self):
        #self.encoder.zero_grad()
        #self.revcoder.zero_grad()
        self.decoder.zero_grad()
        self.W.zero_grad()

    def cuda(self):
        #self.encoder.cuda()
        #self.revcoder.cuda()
        self.decoder.cuda()
        self.W.cuda()

    def getIndex(self,row,inference=False):
        tensor=torch.LongTensor(row)
        if torch.cuda.is_available():
            tensor=tensor.cuda()
        return autograd.Variable(tensor,volatile=inference)

    def init_hidden(self,batch):
        hiddenElem1=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size)
        if self.cnfg.use_LSTM:
            hiddenElem2=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size)
        if torch.cuda.is_available():
            hiddenElem1=hiddenElem1.cuda()
            if self.cnfg.use_LSTM:
                hiddenElem2=hiddenElem2.cuda()
        if self.cnfg.use_LSTM: 
            return (autograd.Variable(hiddenElem1),autograd.Variable(hiddenElem2))
        else:
            return autograd.Variable(hiddenElem1)

    def save_checkpoint(self,modelName,optimizer):
        checkpoint={'decoder_state_dict':self.decoder.state_dict(),'lin_dict':self.W.state_dict(),'optimizer':optimizer.state_dict()} 
        torch.save(checkpoint,self.cnfg.model_dir+modelName+"lm.ckpt")
        print "Saved Model"
        return

    def getParams(self):
        all_params=list(self.decoder.parameters())+list(self.W.parameters())
        return all_params

    def load_from_checkpoint(self,modelName,optimizer=None):
        checkpoint=torch.load(modelName)
        
        #self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        #self.revcoder.load_state_dict(checkpoint['revcoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.W.load_state_dict(checkpoint['lin_dict'])

        if optimizer!=None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print "Loaded Model"
        return

    def decodeAll(self,srcBatches,modelName,method="greedy",evalMethod="BLEU",suffix="test"):
        tgtStrings=[]
        tgtTimes=[]
        totalTime=0.0
        print "Decoding Start Time:",datetime.datetime.now()
        for i,srcBatch in enumerate(srcBatches):
            tgtString=None
            startTime=datetime.datetime.now()
            if method=="greedy":
                tgtString=self.greedyDecode(srcBatch)
            elif method=="beam":
                tgtString=self.beamDecode(srcBatch)
            endTime=datetime.datetime.now()
            timeTaken=(endTime-startTime).total_seconds()
            totalTime+=timeTaken
            if i%100==0:
                print "Decoding Example ",i," Time Taken ",timeTaken
            tgtTimes.append(timeTaken)
            tgtStrings.append(tgtString)
        print "Decoding End Time:",datetime.datetime.now()
        print "Total Decoding Time:",totalTime
        
        #Dump Output
        if method=="greedy":
            outFileName=modelName+"."+suffix+".output"
        else:
            outFileName=modelName+"."+suffix+"."+method+".output"

        outFile=open(outFileName,"w")
        for tgtString in tgtStrings:
            outFile.write(tgtString+"\n")
        outFile.close()

        #Dump Times
        timeFileName=modelName+"."+suffix+".time"
        timeFile=open(timeFileName,"w")
        for tgtTime in tgtTimes:
            timeFile.write(str(tgtTime)+"\n")
        timeFile.close()

        if evalMethod=="BLEU":
            import os
            if self.cnfg.problem=="MT":
                BLEUOutput=os.popen("perl multi-bleu.perl -lc "+"data/"+suffix+".en-de.low.en"+" < "+outFileName).read()
            elif self.cnfg.problem=="CHESS":
                BLEUOutput=os.popen("perl multi-bleu.perl -lc "+"data/"+suffix+".che-eng.single.en"+" < "+outFileName).read()

            print BLEUOutput
        #Compute BLEU
        elif evalMethod=="ROUGE":
            print "To implement ROUGE"

        return tgtStrings

    def beamDecode(self,srcBatch):
        k=self.cnfg.beamSize
        srcSentenceLength=srcBatch.shape[1]
        srcBatch=srcBatch.T
        self.enc_hidden=self.init_hidden(srcBatch)
        enc_out=None
        encoderOuts=[]
        if self.cnfg.use_reverse:
            self.rev_hidden=self.init_hidden(srcBatch)
            rev_out=None
            revcoderOuts=[]
        srcEmbedIndexSeq=[]
        for rowId,row in enumerate(srcBatch):
            srcEmbedIndex=self.getIndex(row,inference=True)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out,self.enc_hidden=self.encoder(srcBatch.shape[1],srcEmbedIndex,self.enc_hidden)
            encoderOuts.append(enc_out.view(1,-1))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out,self.rev_hidden=self.revcoder(srcBatch.shape[1],srcEmbedIndex,self.rev_hidden)
                
                revcoderOuts.append(rev_out.view(1,-1))
            revcoderOuts.reverse()

        
        if self.cnfg.use_reverse:
            encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del srcBatch
            del enc_out

        zeroInit=torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)


        self.hidden=self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed==False:
                if self.cnfg.init_enc:
                    self.hidden=self.enc_hidden
                else:
                    self.hidden=self.rev_hidden
            else:
                if self.cnfg.use_LSTM:
                    self.hidden=(torch.add(self.enc_hidden[0],self.rev_hidden[0]),torch.add(self.enc_hidden[1],self.rev_hidden[1]))
                else:
                    self.hidden=torch.add(self.enc_hidden,self.rev_hidden)
        
        tgts=[] 
        row=np.array([self.cnfg.start,]*1)
        
        tgtEmbedIndex=self.getIndex(row,inference=True)
        
        out,self.hidden,c_0=self.decoder(1,tgtEmbedIndex,None,None,self.hidden,feedContextVector=True,contextVector=c_0)
        #forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
        
        out=out.view(1,-1)
        if self.cnfg.use_attention:
            scores=self.W(torch.cat([out,c_0],1))
        else:
            scores=self.W(out)

        maxValues,argmaxes=torch.max(scores,1)
        argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            if not (self.cnfg.decoder_prev_random or self.cnfg.mixed_decoding):
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        encOutTensor=torch.cat([encoderOut.view(1,1,self.cnfg.hidden_size) for encoderOut in encoderOuts],1)

        beams=[(self.hidden,out,0.0,[tgts[0],],False)] #Current state, current output, current score, current tgts, stopped boolean
        completedBeams=[] #Stop when this reaches k.

        steps=0
        while len(completedBeams)<k and steps<2*srcSentenceLength+10: #self.cnfg.TGT_LEN_LIMIT:
            #print "Step ",steps
            expandedBeams=[]
            for beam in beams:
                if beam[4]:
                    continue
                row=np.array([beam[3][-1],]*1)
                tgtEmbedIndex=self.getIndex(row,inference=True)
                o_t=beam[1] #out
                #print np.shape(row)
                #print tgtEmbedIndex.size()
                #print o_t.size()
                #print beam[0][0].size()
                #print beam[0][1].size()
                #print encOutTensor.size()
                out,newHidden,c_t=self.decoder(1,tgtEmbedIndex,torch.transpose(encOutTensor,0,1),o_t,beam[0],feedContextVector=False,inference=True)

                del o_t

                out=out.view(1,-1)
                if self.cnfg.use_attention:
                    scores=F.log_softmax(self.W(torch.cat([out,c_t],1)))
                else:
                    scores=F.log_softmax(self.W(out))

                maxValues,argmaxes=torch.topk(scores,k=k,dim=1)
                argmaxValues=argmaxes.cpu().data.numpy()
                maxValues=maxValues.cpu().data.numpy()
                for kprime in range(k):
                    argmaxValue=argmaxValues[:,kprime][0]
                    maxValue=maxValues[:,kprime][0]
                    newScore=beam[2]+maxValue-0.5*kprime
                    entry=(newHidden,out,newScore,list(beam[3])+[argmaxValue,],False)
                    if argmaxValue==self.cnfg.stop:
                        modifiedEntry=(newHidden,out,newScore,list(beam[3])+[argmaxValue,],True)
                        completedBeams.append(entry)
                        #print "Completed Beam: ",len(completedBeams)
                    else:
                        expandedBeams.append(entry)
                        #print "Expanded Beam: ",len(expandedBeams)
                #argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
                #tgts.append(argmaxValue)

                #Add to expandedBeams
                #Add stopped beams to completedBeams
            
            expandedBeams.sort(key= lambda x:-x[2]) #Sort the expanded beams
            beams=expandedBeams[:k] #Keep top k for next iteration
            steps+=1
        
        #Put remaining beams into expanded Beams
        
        #Filter out overtly short beams
        newExpandedBeams=[]
        for beam in completedBeams+beams:
            if len(beam[3])>=3:
                newExpandedBeams.append(beam)

        #print "Final Number of Expanded Beams:",len(newExpandedBeams)
        newExpandedBeams.sort(key = lambda x: -x[2]/len(x[3]))
        tgts=newExpandedBeams[0][3]
        if tgts[-1]==self.cnfg.stop:
            tgts=tgts[:-1]

        return " ".join([self.reverse_wids_tgt[x] for x in tgts])




    def greedyDecode(self,srcBatch):
        #Note: srcBatch is of size 1
        srcSentenceLength=srcBatch.shape[1]
        srcBatch=srcBatch.T
        self.enc_hidden=self.init_hidden(srcBatch)
        enc_out=None
        encoderOuts=[]
        if self.cnfg.use_reverse:
            self.rev_hidden=self.init_hidden(srcBatch)
            rev_out=None
            revcoderOuts=[]

        srcEmbedIndexSeq=[]
        for rowId,row in enumerate(srcBatch):
            srcEmbedIndex=self.getIndex(row,inference=True)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out,self.enc_hidden=self.encoder(srcBatch.shape[1],srcEmbedIndex,self.enc_hidden)
            encoderOuts.append(enc_out.view(1,-1))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out,self.rev_hidden=self.revcoder(srcBatch.shape[1],srcEmbedIndex,self.rev_hidden)
                
                revcoderOuts.append(rev_out.view(1,-1))
            revcoderOuts.reverse()

        
        if self.cnfg.use_reverse:
            encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del srcBatch
            del enc_out

        zeroInit=torch.zeros(encoderOuts[-1].size())
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)


        self.hidden=self.enc_hidden
        if self.cnfg.use_reverse:
            if self.cnfg.init_mixed==False:
                if self.cnfg.init_enc:
                    self.hidden=self.enc_hidden
                else:
                    self.hidden=self.rev_hidden
            else:
                if self.cnfg.use_LSTM:
                    self.hidden=(torch.add(self.enc_hidden[0],self.rev_hidden[0]),torch.add(self.enc_hidden[1],self.rev_hidden[1]))
                else:
                    self.hidden=torch.add(self.enc_hidden,self.rev_hidden)
        
        tgts=[] 
        row=np.array([self.cnfg.start,]*1)
        
        tgtEmbedIndex=self.getIndex(row,inference=True)
        
        out,self.hidden,c_0=self.decoder(1,tgtEmbedIndex,None,None,self.hidden,feedContextVector=True,contextVector=c_0)
        #forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
        
        out=out.view(1,-1)
        if self.cnfg.use_attention:
            scores=self.W(torch.cat([out,c_0],1))
        else:
            scores=self.W(out)

        maxValues,argmaxes=torch.max(scores,1)
        argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
        tgts.append(argmaxValue)

        if self.cnfg.mem_optimize:
            if not (self.cnfg.decoder_prev_random or self.cnfg.mixed_decoding):
                del c_0
            del self.enc_hidden
            if self.cnfg.use_reverse:
                del self.rev_hidden

        encOutTensor=torch.cat([encoderOut.view(1,1,self.cnfg.hidden_size) for encoderOut in encoderOuts],1)
        while argmaxValue!=self.cnfg.stop and len(tgts)<2*srcSentenceLength+10: #self.cnfg.TGT_LEN_LIMIT:
            row=np.array([argmaxValue,]*1)
            tgtEmbedIndex=self.getIndex(row,inference=True)
            o_t=out

            out,self.hidden,c_t=self.decoder(1,tgtEmbedIndex,torch.transpose(encOutTensor,0,1),o_t,self.hidden,feedContextVector=False,inference=True)

            del o_t

            out=out.view(1,-1)
            if self.cnfg.use_attention:
                scores=self.W(torch.cat([out,c_t],1))
            else:
                scores=self.W(out)

            maxValues,argmaxes=torch.max(scores,1)
            argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
            tgts.append(argmaxValue)

        if tgts[-1]==self.cnfg.stop:
            tgts=tgts[:-1]

        return " ".join([self.reverse_wids_tgt[x] for x in tgts])

    
    def forward(self,batch,mask,loss_function,inference=False):
        """
        srcBatch=srcBatch.T
        srcMask=srcMask.T
        #Init encoder. We don't need start here since we don't softmax.
        self.enc_hidden=self.init_hidden(srcBatch)
        #print "Src Batch Size:",srcBatch.shape
        #print "Src Mask Size:",srcMask.shape

        enc_out=None
        encoderOuts=[]

        if self.cnfg.use_reverse:
            self.rev_hidden=self.init_hidden(srcBatch)
            rev_out=None
            revcoderOuts=[]

        srcEmbedIndexSeq=[]
        for rowId,row in enumerate(srcBatch):
            srcEmbedIndex=self.getIndex(row,inference=inference)
            if self.cnfg.use_reverse:
                srcEmbedIndexSeq.append(srcEmbedIndex)

            enc_out,self.enc_hidden=self.encoder(srcBatch.shape[1],srcEmbedIndex,self.enc_hidden)

            encoderOuts.append(enc_out.squeeze(0))

        if self.cnfg.use_reverse:
            srcEmbedIndexSeq.reverse()
            for srcEmbedIndex in srcEmbedIndexSeq:
                rev_out,self.rev_hidden=self.revcoder(srcBatch.shape[1],srcEmbedIndex,self.rev_hidden)
                revcoderOuts.append(rev_out.squeeze(0))
            revcoderOuts.reverse()

        if self.cnfg.use_reverse:
            encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if self.cnfg.srcMasking:
            srcMaskTensor=torch.Tensor(srcMask)
            if torch.cuda.is_available():
                srcMaskTensor=srcMaskTensor.cuda()
            srcMaskTensor=torch.chunk(autograd.Variable(srcMaskTensor),len(encoderOuts),0)
            srcMaskTensor=[x.view(-1,1) for x in srcMaskTensor]
            encoderOuts=[encoderOut*(x.expand(encoderOut.size())) for encoderOut,x in zip(encoderOuts,srcMaskTensor)]
            del srcMaskTensor

        if self.cnfg.mem_optimize:
            if self.cnfg.use_reverse:
                del revcoderOuts
                del rev_out
            del srcEmbedIndexSeq
            del srcBatch
            del enc_out
        """
 
        batch=batch.T        
        self.hidden=self.init_hidden(batch)
        
        row=np.array([self.cnfg.start,]*batch.shape[1])
        
        tgtEmbedIndex=self.getIndex(row,inference=inference)
        #forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
        out,self.hidden=self.decoder(batch.shape[1],tgtEmbedIndex,self.hidden)
       
        decoderOuts=[out.squeeze(0),]
        tgts=[]
        for rowId,row in enumerate(batch):
            
            tgtEmbedIndex=self.getIndex(row,inference=inference)
            o_t=decoderOuts[-1]

            #forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None)
            out,self.hidden=self.decoder(batch.shape[1],tgtEmbedIndex,self.hidden)

            tgts.append(self.getIndex(row))
            decoderOuts.append(out.squeeze(0))


        decoderOuts=decoderOuts[:-1]

        if self.cnfg.mem_optimize:
            del out
            del self.hidden
            gc.collect()

        totalLoss=sum([loss_function(F.log_softmax(self.W(decoderOut)),tgt) for decoderOut,tgt in zip(decoderOuts,tgts)])
        return totalLoss
