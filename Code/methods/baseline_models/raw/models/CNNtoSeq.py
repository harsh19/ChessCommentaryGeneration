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

class CNNSeqToSeqAttn(nn.Module):
    def __init__(self,cnfg,wids_src=None,wids_tgt=None,encoder_configurations=['cnn','cnn', 'birnn'],typ="normal"): # As of now assumes shared vocab on all input sequences
        super(CNNSeqToSeqAttn,self).__init__()
        self.wids_src=wids_src
        self.wids_tgt=wids_tgt
        self.reverse_wids_src=torch_utils.reverseDict(wids_src)
        self.reverse_wids_tgt=torch_utils.reverseDict(wids_tgt)
        self.cnfg=cnfg
        self.cnfg.srcVocabSize=len(self.wids_src)
        self.cnfg.tgtVocabSize=len(self.wids_tgt)
        self.models={}
        self.configurations={}
        self.typ=typ

	self.hidden_size = cnfg.hidden_size
        self.src_hidden_size = 64 #self.cnfg.hidden_size
        src_emb_size = self.src_emb_size = 32 #cnfg.emb_size
        self.emb_size = cnfg.emb_size
        output_channels =  self.output_channels = 40 #20 # 10
        kernel_size=2 # 2*2
        self.cnn_output_size = cnn_output_size = 4 #16 ### not a param for cnn. output of cnn is batchsize, outputchannels, cnn_output_size. cnn_output_size is used in later modules

        # Encoder
        self.encoder_configurations = encoder_configurations
        self.encoder_embeddings=nn.Embedding(self.cnfg.srcVocabSize, src_emb_size) # source side embeddigs
        s="encoder_prevboard"
        self.encoder_prevboard = CNN( inp_channels=src_emb_size, output_channels=output_channels, kernel_size=kernel_size)
        s="encoder_curboard"
        self.encoder_curboard = CNN( inp_channels=src_emb_size, output_channels=output_channels, kernel_size=kernel_size)
        s="encoder_move"
        self.encoder_move=EncoderRNNModel(self.wids_src,self.cnfg.srcVocabSize,src_emb_size,self.src_hidden_size,self.cnfg.use_LSTM,True,self.encoder_embeddings)

        # Decoder
        self.decoder=AttnDecoderRNNMutipleInputCNN(self.wids_tgt,cnfg.tgtVocabSize,cnfg.emb_size,cnfg.hidden_size,cnfg.use_LSTM,cnfg.use_attention,cnfg.share_embeddings,reference_embeddings=None,extra_dims=self.src_hidden_size,cnn_output_size=cnn_output_size,typ=self.typ)
        if self.cnfg.use_attention and self.cnfg.use_downstream:
            additional = self.src_hidden_size + 2*self.cnn_output_size
            self.W=LinearLayer(additional + self.cnfg.hidden_size,self.cnfg.tgtVocabSize)
        else:
            self.W=LinearLayer(self.cnfg.hidden_size,self.cnfg.tgtVocabSize)

        # print trainable variables
        print "CNN to seq:", self._modules.keys()
        print "------------------------"
        for param in self.parameters():
            print(type(param.data), param.size())
        print "------------------------"


    def getIndex(self,row,inference=False):
        tensor=torch.LongTensor(row)
        if torch.cuda.is_available():
            tensor=tensor.cuda()
        return autograd.Variable(tensor,volatile=inference)

    def init_hidden(self,batch):
        # batch -> seqlen, b
        hiddenElem1=torch.zeros(1,batch.shape[1],self.src_hidden_size) # 1,b,hidden_size
        hiddenElem2=torch.zeros(1,batch.shape[1],self.src_hidden_size) # (1,b,hidden_size),(1,b,hidden_size)
        if torch.cuda.is_available():
            hiddenElem1=hiddenElem1.cuda()
            hiddenElem2=hiddenElem2.cuda()
        return (autograd.Variable(hiddenElem1),autograd.Variable(hiddenElem2))

    def save_checkpoint(self,modelName,optimizer):
        checkpoint={k:v.state_dict() for k,v in self._modules.items()}
        checkpoint['optimizer']=optimizer.state_dict()
        torch.save(checkpoint,self.cnfg.model_dir+modelName+".ckpt")
        print "Saved Model"

    def getParams(self):
        return self.parameters()

    def load_from_checkpoint(self,modelName,optimizer=None): #TODO
        checkpoint=torch.load(modelName)
        for k,model in self._modules:
            model.load_state_dict(checkpoint[k])
        if optimizer!=None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print "Loaded Model"

    def decodeAll(self,src_batches,src_masks,modelName,method="greedy",evalMethod="BLEU",suffix="test"):
        tgtStrings=[]
        tgtTimes=[]
        totalTime=0.0
        print "Decoding Start Time:",datetime.datetime.now()
        batch_indices = range(len(src_batches[0]))

        for batchId in batch_indices:

            src_batch = [ele[batchId] for ele in src_batches]
            src_mask = [ele[batchId] for ele in src_masks]
            batchLength=src_batch[0].shape[1] # src_batch[0] corresponds to prevboard
            batchSize=src_batch[0].shape[0] # 1

            tgtString=None
            startTime=datetime.datetime.now()
            if method=="greedy":
                tgtString=self.forward(srcBatchList=src_batch,srcMaskList=src_mask,mode="decode_greedy")
            endTime=datetime.datetime.now()
            timeTaken=(endTime-startTime).total_seconds()
            totalTime+=timeTaken
            if batchId%100==0:
                print "Decoding Example ",batchId," Time Taken ",timeTaken
            tgtTimes.append(timeTaken)
            tgtStrings.append(tgtString)

        print "Decoding End Time:",datetime.datetime.now()
        print "Total Decoding Time:",totalTime

        #Dump Output
        outFileName=modelName+"."+suffix+".output"
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
            BLEUOutput=os.popen("perl multi-bleu.perl -lc "+"data/"+suffix+".che-eng.single.en"+" < "+outFileName).read()
            print BLEUOutput
        #Compute BLEU
        elif evalMethod=="ROUGE":
            print "To implement ROUGE"

        return tgtStrings


    def _initDecoder(self, encoder_hidden_to_use, init_size, init_with_zero=False,init_with_zero_size=-1):
        zeroInit=torch.zeros(init_size)
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)
	hidden = None
	if init_with_zero:
		hidden0 =  autograd.Variable(torch.zeros(init_with_zero_size))
		hidden1 =  autograd.Variable(torch.zeros(init_with_zero_size))
		if torch.cuda.is_available():
			hidden0 = hidden0.cuda()
			hidden1 = hidden1.cuda()
		hidden= hidden0, hidden1
		#hidden = autograd.Variable(hidden)
	else:
	        hidden = encoder_hidden_to_use #
        return c_0, hidden

    def _decoderStep(self, batch_size, cur_inputs, encoder_outputs, o_t, previous_hidden,inference=False):
        tgtEmbedIndex=self.getIndex(cur_inputs,inference=inference)
        out,hidden,c_t=self.decoder(batch_size,tgtEmbedIndex,encoder_outputs,o_t,previous_hidden,feedContextVector=False,inference=inference)
        return out,hidden,c_t

    def forward(self, srcBatchList, trgtBatch=None, srcMaskList=None, trgtMask=None, loss_function=None, inference=False, mode="train"): # mode: train, greedy_decode, beam_decode

        batch_size = srcBatchList[0].shape[0]

        # encoding
        prevboard = srcBatchList[0]
        curboard = srcBatchList[1]
        move = srcBatchList[2]
        enc_outs=[]
        hidden_vals=[]

        #### prevboard
        #prevboard = prevboard.view(batch_size,-1,8,8)
        prevboard_index = self.getIndex(prevboard,inference=inference)
        embeddings_lookup = self.encoder_embeddings(prevboard_index)
        embedding_mask = torch.FloatTensor( np.expand_dims(srcMaskList[0],2) )
        if torch.cuda.is_available():
            embedding_mask = embedding_mask.cuda()
        embedding_mask = autograd.Variable(embedding_mask,requires_grad=False)
	#print embeddings_lookup.data.shape, embedding_mask.data.shape
	embeddings_lookup = embeddings_lookup * embedding_mask  #srcMaskList[0].expand_dims( 2 )  #TODO revert back the added dimensions embeddings_lookup.size() )  #embeddings_lookup.view(-1,self.src_emb_size) # b*64,1 and b*64,emb => b*64, emb
	#print embeddings_lookup.data.shape, embeddings_lookup.data.shape
        ## TODO  = srcMaskList[0].reshape(-1,1) * embeddings_lookup # b*64,1 and b*64,emb => b*64, emb
        embeddings_lookup = embeddings_lookup.view(batch_size,8,8,self.src_emb_size)
        embeddings_lookup = embeddings_lookup.permute(0,3,1,2) # change to NCHW from NHWC
        #print "embeddings_lookup.data.shape: ",embeddings_lookup.data.shape
        cnn_linear_rep_prevboard = self.encoder_prevboard(embeddings_lookup) # batchsize, outputchannels, outputsize
        cnn_linear_rep_prevboard = cnn_linear_rep_prevboard.permute(1,0,2) # outputchannels, batchsize, outputsize

        #### curboard
        curboard_index = self.getIndex(curboard,inference=inference)
        embeddings_lookup = self.encoder_embeddings(curboard_index)
        ## TODO embeddings_lookup = srcMaskList[1].view(-1,1) * embeddings_lookup.view(-1,self.src_emb_size) # b*64,1 and b*64,emb => b*64, emb
        embedding_mask = torch.FloatTensor( np.expand_dims(srcMaskList[1],2) )
        if torch.cuda.is_available():
            embedding_mask = embedding_mask.cuda()
        embedding_mask = autograd.Variable(embedding_mask,requires_grad=False)
	embeddings_lookup = embeddings_lookup * embedding_mask #srcMaskList[1].expand_dims(2 ) # embeddings_lookup.size() )  #embeddings_lookup.view(-1,self.src_emb_size) # b*64,1 and b*64,emb => b*64, emb
        embeddings_lookup = embeddings_lookup.view(batch_size,8,8,self.src_emb_size)
        embeddings_lookup = embeddings_lookup.permute(0,3,1,2) # change to NCHW from NHWC
        cnn_linear_rep_curboard = self.encoder_curboard(embeddings_lookup)
        cnn_linear_rep_curboard = cnn_linear_rep_curboard.permute(1,0,2)

        #### move
        hidden=self.init_hidden(move.T) # enc_hidden-> b,enc_hidden_size
        enc_out, hidden_val = self.encoder_move( move.T, srcMaskList[2].T, hidden=hidden , inference=inference, srcMasking=self.cnfg.srcMasking)
        #print "enc out shape : ", enc_out.data.shape
        hidden_vals = [hidden_val]
        ######## Add all
        enc_outs = [cnn_linear_rep_prevboard,cnn_linear_rep_curboard, enc_out]
        #print "*----------------------------------"

        # init decoding
        c_0,hidden = None,None
        #print " --- ",enc_outs[-1][-1].size()
        c_0,hidden = self._initDecoder(hidden_vals[-1], (batch_size, self.src_hidden_size+2*self.cnn_output_size), True, (1,batch_size, self.hidden_size) ) # init with move
        #c_0,hidden = self._initDecoder(hidden_vals[-1], (batch_size, self.src_hidden_size+2*self.cnn_output_size) ) # init with move
        #c_0,hidden = self._initDecoder(hidden_vals[-1], enc_outs[-1][-1].size() ) # init with move

        #Decoding
        if self.cnfg.use_attention:
            contextVectors=[]
            contextVectors.append(c_0)
        row=np.array([self.cnfg.start,]*srcBatchList[0].shape[0]) # b,1
        #row=np.array([self.cnfg.start,]*trgtBatch.shape[1]) # b,1

        encoder_outs_tensors=[]
        for encoder_outs_j in enc_outs: # for each input seq
            encoderOutTensor=torch.stack([encoderOut for encoderOut in encoder_outs_j],dim=0)
            encoder_outs_tensors.append(encoderOutTensor)
        # encoder_outs_tensors: num_input_sequences * seq length * b * hidden_size

        encoder_outs_combined = encoder_outs_tensors
        decoderOuts= [row,] #[row,] #[out.squeeze(0),]
        tgtEmbedIndex=self.getIndex(row,inference=inference) # b,1
        #print "tgtEmbedIndex = ",tgtEmbedIndex
        out,hidden,c_0=self.decoder(batch_size,tgtEmbedIndex,None,None,hidden,feedContextVector=True,contextVector=c_0)
        decoderOuts=[out.squeeze(0),]
        tgts=[]

        if mode=="train":

            trgtBatch=trgtBatch.T # seqlen,b

            for rowId,row in enumerate(trgtBatch):
                o_t=decoderOuts[-1]
                #print o_t.data.shape, hidden[0].data.shape, row.shape
                out,hidden,c_t = self._decoderStep(batch_size, row, encoder_outs_combined, o_t, hidden)
                tgts.append(self.getIndex(row))
                decoderOuts.append(out.squeeze(0))
                contextVectors.append(c_t)
                #print o_t.data.shape

            decoderOuts=decoderOuts[:-1]
            if self.cnfg.use_attention:
                contextVectors=contextVectors[:-1]
            if self.cnfg.use_attention and self.cnfg.use_downstream:
                decoderOuts=[torch.cat([decoderOut,c_t],1) for decoderOut,c_t in zip(decoderOuts,contextVectors)]
            totalLoss=sum([loss_function(F.log_softmax(self.W(decoderOut)),tgt) for decoderOut,tgt in zip(decoderOuts,tgts)])

            return totalLoss

        else:

            out = out.squeeze(0)

            if self.cnfg.use_attention:
                scores=self.W(torch.cat([out,c_0],1))
            else:
                scores=self.W(out) # to check

            maxValues,argmaxes=torch.max(scores,1)
            argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
            tgts.append(argmaxValue)

            srcSentenceLength = srcBatchList[0].shape[1]

            while argmaxValue!=self.cnfg.stop and len(tgts)<2*srcSentenceLength+10: #self.cnfg.TGT_LEN_LIMIT:

                row=np.array([argmaxValue,]*1)
                o_t=out #.squeeze(0)
                out,hidden,c_t = self._decoderStep(batch_size, row, encoder_outs_combined, o_t, hidden, inference=True)
                #out=out.view(1,-1)
                out = out.squeeze(0)

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
