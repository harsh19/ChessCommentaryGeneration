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
from utils.beam_search import *

class MultiSeqToSeqAttn(nn.Module):
    def __init__(self,cnfg,wids_src=None,wids_tgt=None,encoder_configurations=['birnn'],typ="two_tuple"): # As of now assumes shared vocab on all input sequences
        super(MultiSeqToSeqAttn,self).__init__()
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

        shared_embeddings =nn.Embedding(self.cnfg.srcVocabSize,self.cnfg.emb_size)
        for j,encoder_config in enumerate(encoder_configurations):
            s="encoder_"+str(j)
            if encoder_config=="birnn":
                encoder=EncoderBiRNNModel(self.wids_src,self.cnfg.srcVocabSize,self.cnfg.emb_size,self.cnfg.hidden_size,self.cnfg.use_LSTM,True,shared_embeddings)
            elif encoder_config=="rnn":
                encoder=EncoderRNNModel(self.wids_src,self.cnfg.srcVocabSize,self.cnfg.emb_size,self.cnfg.hidden_size,self.cnfg.use_LSTM,True,shared_embeddings)
            self.configurations[s]=encoder_config
            self.models[s]=encoder
            self.add_module(s,encoder)

        self.decoder=AttnDecoderRNNMutipleInput(self.wids_tgt,cnfg.tgtVocabSize,cnfg.emb_size,cnfg.hidden_size,cnfg.use_LSTM,cnfg.use_attention,cnfg.share_embeddings,reference_embeddings=shared_embeddings,extra_dims=self.cnfg.hidden_size,typ=self.typ)
        self.models['decoder']=self.decoder

        if self.cnfg.use_attention and self.cnfg.use_downstream:
            self.W=LinearLayer(2*self.cnfg.hidden_size,self.cnfg.tgtVocabSize)
        else:
            self.W=LinearLayer(self.cnfg.hidden_size,self.cnfg.tgtVocabSize)
        self.models['W']=self.W

        print "MultiSeqToSeqAttn:", self._modules.keys()
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
        hiddenElem1=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size) # 1,b,hidden_size
        if self.cnfg.use_LSTM:
            hiddenElem2=torch.zeros(1,batch.shape[1],self.cnfg.hidden_size) # (1,b,hidden_size),(1,b,hidden_size)
        if torch.cuda.is_available():
            hiddenElem1=hiddenElem1.cuda()
            if self.cnfg.use_LSTM:
                hiddenElem2=hiddenElem2.cuda()
        if self.cnfg.use_LSTM:
            return (autograd.Variable(hiddenElem1),autograd.Variable(hiddenElem2))
        else:
            return autograd.Variable(hiddenElem1)

    def save_checkpoint(self,modelName,optimizer):
        checkpoint={k:v.state_dict() for k,v in self.models.items()}
        checkpoint['optimizer']=optimizer.state_dict()
        torch.save(checkpoint,self.cnfg.model_dir+modelName+".ckpt")
        print "Saved Model"

    def getParams(self):
        return self.parameters()

    def load_from_checkpoint(self,modelName,optimizer=None): #TODO
        checkpoint=torch.load(modelName)
        for k,model in self.models.items():
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
                tgtString=self.forward(srcBatchList=src_batch,srcMaskList=src_mask,mode="beam_decode"  )#"beam_decode")
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


    def _initDecoder(self, encoder_hidden_to_use, init_size):
        zeroInit=torch.zeros(init_size)
        if torch.cuda.is_available():
            zeroInit=zeroInit.cuda()
        c_0=autograd.Variable(zeroInit)
        hidden = encoder_hidden_to_use # hidden_curboard + hidden_prevboard
        return c_0, hidden

    def _decoderStep(self, batch_size, cur_inputs, encoder_outputs, o_t, previous_hidden,inference=False):
        tgtEmbedIndex=self.getIndex(cur_inputs,inference=inference)
        print "previous_hidden = ",previous_hidden[0].data.shape
        print "cur_inputs = ",cur_inputs.shape
        print "encoder_outputs = ",encoder_outputs.data.shape
        print "o_t = ",o_t.data.shape
        print "batch_size = ",batch_size
        out,hidden,c_t=self.models['decoder'](batch_size,tgtEmbedIndex,encoder_outputs,o_t,previous_hidden,feedContextVector=False,inference=inference)
        return out,hidden,c_t

    def forward(self,srcBatchList,trgtBatch=None,srcMaskList=None,trgtMask=None,loss_function=None,inference=False,mode="train"): # mode: train, greedy_decode, beam_decode

        batch_size = srcBatchList[0].shape[0]

        # encoding
        enc_outs=[]
        hidden_vals=[]
        for j in range(len(srcBatchList)):  # each is b,seqlen
            s="encoder_"+str(j)
            hidden=self.init_hidden(srcBatchList[j].T) # enc_hidden-> b,enc_hidden_size
            enc_out, hidden_val = self.models[s]( srcBatchList[j].T, srcMaskList[j].T, hidden=hidden , inference=inference, srcMasking=self.cnfg.srcMasking)
            enc_outs.append(enc_out)
            if self.configurations[s].count("bi")>0:
                enc_hidden,rev_hidden=hidden_val
                hidden_val = (torch.add(enc_hidden[0],rev_hidden[0]),torch.add(enc_hidden[1],rev_hidden[1]))
            hidden_vals.append(hidden_val)

        # init decoding
        c_0,hidden = None,None
        if self.type=="three_tuple":
            c_0,hidden = self._initDecoder(hidden_vals[-1], enc_outs[-1][-1].size()) # init with move
        else:
            c_0,hidden = self._initDecoder(hidden_vals[0], enc_outs[0][-1].size())

        ## trgtBatch=trgtBatch.T # seqlen,b
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

        if self.typ=="two_tuple" or self.typ=="three_tuple": # for 3-tuple, only first two (boards) are used. third (move) is used to init decoder instead
            encoder_outs_combined = self.models['decoder'].combineEncoderOutputs(encoder_outs_tensors) # probably: seq length * b * hidden_size
        elif self.typ=="entire_as_tuple":
            encoder_outs_combined=encoder_outs_tensors[0]
        else:
            print "======= NOT SUPPORTED YET ======"

        tgtEmbedIndex=self.getIndex(row,inference=inference) # b,1
        #print "tgtEmbedIndex = ",tgtEmbedIndex
        out,hidden,c_0=self.models['decoder'](batch_size,tgtEmbedIndex,None,None,hidden,feedContextVector=True,contextVector=c_0)
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

        elif mode=="greedy_decode":

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

        elif mode=="beam_decode":

            length_normalization_factor = 0.0
            beam_size = 5

            # batch_size, row, encoder_outs_combined, o_t, hidden
            out = out.squeeze(0)
            if self.cnfg.use_attention:
                scores=self.W(torch.cat([out,c_0],1))
            else:
                scores=self.W(out) # to check
            maxValues,argmaxes=torch.max(scores,1)
            argmaxValue=argmaxes.view(1).cpu().data.numpy()[0]
            tgts.append(argmaxValue)
            srcSentenceLength = srcBatchList[0].shape[1]

            #print "hidden shape = ", hidden
            initial_beam = Commentary(
                sentence=[self.cnfg.start], #argmax instead ?
                state=(hidden[0], hidden[1]),
                logprob=0.0,
                score=0.0,
                metadata=[""])
            partial_captions = TopN(beam_size)
            partial_captions.push(initial_beam)
            complete_captions = TopN(beam_size)

            for _ in range(2*srcSentenceLength+10): #self.cnfg.TGT_LEN_LIMIT:
                encoder_outs_combined_beam = []
                row_beam = []
                o_t_beam = []
                hidden_beam0 = []
                hidden_beam1 = []
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()

                for partial_caption in partial_captions_list:
                    encoder_outs_combined_beam.append(encoder_outs_combined)
                    hidden_beam0.append( partial_caption.state[0] )
                    hidden_beam1.append( partial_caption.state[1] )
                    row_beam.append(partial_caption.sentence[-1])
                    o_t_beam.append(partial_caption.state[1]) # should this be 0 or 1 ?

                row_beam = np.array(row) #self.getIndex(row, inference=True) #np.array(row)
                encoder_outs_combined_beam = torch.stack(encoder_outs_combined_beam).squeeze(0)
                o_t_beam = torch.stack(o_t_beam).squeeze(0)
                hidden_beam0 = torch.stack(hidden_beam0).squeeze(0)
                hidden_beam0 = hidden_beam0.view(1,len(partial_captions_list),-1)
                hidden_beam1 = torch.stack(hidden_beam1).squeeze(0)
                hidden_beam1 = hidden_beam1.view(1,len(partial_captions_list),-1)
                hidden_beam = (hidden_beam0, hidden_beam1)
                print "row_beam.shape[0] : ",row_beam.shape[0]
                out,hidden,context_vectors = self._decoderStep( row_beam.shape[0], row_beam, encoder_outs_combined_beam, o_t_beam, hidden_beam, inference=True)
                out = out.squeeze(0)
                if self.cnfg.use_attention:
                    scores=self.W(torch.cat([out,context_vectors],1))
                else:
                    scores=self.W(out)

                for i, partial_caption in enumerate(partial_captions_list):
                    word_probabilities = scores[i]
                    state = (hidden[i][0], hidden[i][1])
                    # For this partial caption, get the beam_size most probable next words.
                    idxs = np.argpartition(word_probabilities, -beam_size)[-beam_size:]
                    # Each next word gives a new partial caption.
                    for w in idxs:
                        p = word_probabilities[w]
                        if p < 1e-12:
                            continue  # Avoid log(0).
                        sentence = partial_caption.sentence + [w]
                        logprob = partial_caption.logprob + np.log(p)
                        #alphas_list = partial_caption.alphas + [context_vectors[i]]
                        score = logprob
                        if w == self.cnfg.stop:
                            if length_normalization_factor > 0:
                                score /= len(sentence)**length_normalization_factor
                            beam = Caption(sentence, state, logprob, score, metadata=[""])
                            complete_captions[partial_caption_indices[i]].push(beam)
                        else:
                            beam = Caption(sentence, state, logprob, score, metadata=[""])
                            partial_captions.push(beam)
                if partial_captions.size() == 0:
                    # We have run out of partial candidates; happens when beam_size = 1.
                    break

            return " ".join([self.reverse_wids_tgt[x] for x in tgts])

        else:
            print "---- Invalid mode----"
