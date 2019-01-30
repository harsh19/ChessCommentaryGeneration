import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

import torch_utils

import numpy as np
import random
import math
import datetime
import gc
import sys

class EncoderRNN(nn.Module):
    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings,reference_embeddings=None):
        super(EncoderRNN,self).__init__()
        self.vocabSize=vocabSize
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.use_LSTM=use_LSTM
        self.wids=wids
        if share_embeddings:
            self.embeddings=reference_embeddings
        else:
            #print "ok touched here"
            self.embeddings=nn.Embedding(self.vocabSize,self.emb_size)
        if self.use_LSTM:
            self.encoder=nn.LSTM(self.emb_size,self.hidden_size)
        else:
            self.encoder=nn.GRU(self.emb_size,self.hidden_size)
        print "encoder rnn: ",self._modules.keys()

    def forward(self,batchSize,embedIndex,hidden):
        embeds=self.embeddings(embedIndex)
        #print hidden[0].data.shape, embeds.data.shape
	out,hidden=self.encoder(embeds.view(1,batchSize,-1),hidden)
        return out,hidden


def getIndex(row,inference=False):
    tensor=torch.LongTensor(row)
    if torch.cuda.is_available():
        tensor=tensor.cuda()
    return autograd.Variable(tensor,volatile=inference)


class EncoderBiRNNModel(nn.Module):

    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings,reference_embeddings=None):
        super(EncoderBiRNNModel,self).__init__()
        self.encoder=EncoderRNN(wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings, reference_embeddings)
        if not share_embeddings:
            reference_embeddings=self.encoder.embeddings
        self.revcoder=EncoderRNN(wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings=True, reference_embeddings=reference_embeddings)
        print "EncoderBiRNNModel: ",self._modules.keys()


    def forward(self,srcBatch,srcMask,hidden,srcMasking=False,inference=False):
        encoderOuts = []
        srcEmbedIndexSeq=[]
        enc_hidden, rev_hidden = hidden, hidden
        for rowId,row in enumerate(srcBatch): ## seqlen times. row: b
            srcEmbedIndex=getIndex(row,inference=inference) # b, embedding_size
            srcEmbedIndexSeq.append(srcEmbedIndex)
            enc_out,enc_hidden=self.encoder(srcBatch.shape[1],srcEmbedIndex,enc_hidden) # encoder( value b, (b,embedding_size), b,enc_hidden_size ) .  return is enc_out,enc_hidden.
            encoderOuts.append(enc_out.squeeze(0)) # enc_out: 1,b,hidden_size. After squeeze: b,hidden_size
        # encoderOuts-> seqlen,b,hidden_size

        revcoderOuts=[]
        srcEmbedIndexSeq.reverse()
        for srcEmbedIndex in srcEmbedIndexSeq:
            rev_out,rev_hidden=self.revcoder(srcBatch.shape[1],srcEmbedIndex,rev_hidden)
            revcoderOuts.append(rev_out.squeeze(0))
        revcoderOuts.reverse()

        encoderOuts=[torch.add(x,y) for x,y in zip(encoderOuts,revcoderOuts)]

        if srcMasking:
            srcMaskTensor=torch.Tensor(srcMask)
            if torch.cuda.is_available():
                srcMaskTensor=srcMaskTensor.cuda()
            srcMaskTensor=torch.chunk(autograd.Variable(srcMaskTensor),len(encoderOuts),0)
            srcMaskTensor=[x.view(-1,1) for x in srcMaskTensor]
            encoderOuts=[encoderOut*(x.expand(encoderOut.size())) for encoderOut,x in zip(encoderOuts,srcMaskTensor)]
            del srcMaskTensor

        return encoderOuts, (enc_hidden, rev_hidden)


class EncoderRNNModel(nn.Module):

    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings,reference_embeddings=None):
        super(EncoderRNNModel,self).__init__()
        self.encoder=EncoderRNN(wids,vocabSize,emb_size,hidden_size,use_LSTM,share_embeddings, reference_embeddings)

    def forward(self,srcBatch,srcMask,hidden,srcMasking=False,inference=False):
        encoderOuts = []
        enc_hidden = hidden
        for rowId,row in enumerate(srcBatch): ## seqlen times. row: b
            srcEmbedIndex=getIndex(row,inference=inference) # b, embedding_size
            enc_out,enc_hidden=self.encoder(srcBatch.shape[1],srcEmbedIndex,enc_hidden) # encoder( value b, (b,embedding_size), b,enc_hidden_size ) .  return is enc_out,enc_hidden.
            encoderOuts.append(enc_out.squeeze(0)) # enc_out: 1,b,hidden_size. After squeeze: b,hidden_size
        # encoderOuts-> seqlen,b,hidden_size

        if srcMasking:
            srcMaskTensor=torch.Tensor(srcMask)
            if torch.cuda.is_available():
                srcMaskTensor=srcMaskTensor.cuda()
            srcMaskTensor=torch.chunk(autograd.Variable(srcMaskTensor),len(encoderOuts),0)
            srcMaskTensor=[x.view(-1,1) for x in srcMaskTensor]
            encoderOuts=[encoderOut*(x.expand(encoderOut.size())) for encoderOut,x in zip(encoderOuts,srcMaskTensor)]
            del srcMaskTensor

        return encoderOuts, enc_hidden

################################################################

class AttnDecoderRNN(nn.Module):
    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,use_attention,share_embeddings,reference_embeddings=None,sigmoid=False):
        super(AttnDecoderRNN,self).__init__()
        self.vocabSize=vocabSize
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.use_LSTM=use_LSTM
        self.use_attention=use_attention
        self.wids=wids
        self.sigmoid=sigmoid

        if share_embeddings:
            self.embeddings=reference_embeddings
        else:
            self.embeddings=nn.Embedding(self.vocabSize,self.emb_size)

        if self.use_attention:
            if self.use_LSTM:
                self.decoder=nn.LSTM(self.emb_size+self.hidden_size,self.hidden_size)
            else:
                self.decoder=nn.GRU(self.emb_size+self.hidden_size,self.hidden_size)
        else:
            if self.use_LSTM:
                self.decoder=nn.LSTM(self.emb_size,self.hidden_size)
            else:
                self.decoder=nn.GRU(self.emb_size,self.hidden_size)

    def forward(self,batchSize,tgtEmbedIndex,encoderOutTensor,o_t,hidden,feedContextVector=False,contextVector=None,inference=False,getAtt=False):

        if self.use_attention:
            if not feedContextVector:
                o_t_expanded=o_t.expand(encoderOutTensor.size())
                dotProduct=torch.transpose(torch.sum(torch.mul(encoderOutTensor,o_t_expanded),2),0,1)
                del o_t_expanded
                if not getAtt:
                    if not self.sigmoid:
                        alphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
                    else:
                        alphas=torch.transpose(F.sigmoid(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
                else:
                    if not self.sigmoid:
                        firstAlphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2)
                    else:
                        firstAlphas=torch.transpose(F.sigmoid(dotProduct),0,1).unsqueeze(2)
                    alphas=firstAlphas.expand(encoderOutTensor.size())
                    alphasNumpy=firstAlphas.data.cpu().numpy()
                del o_t
                c_t=torch.squeeze(torch.sum(alphas*encoderOutTensor,0),0)
            else:
                c_t=contextVector


        tgtEmbeds=self.embeddings(tgtEmbedIndex)
        if inference:
            c_t=c_t.view(1,-1)
        if self.use_attention:
            tgtEmbeds=torch.cat([tgtEmbeds,c_t],1).view(1,batchSize,-1)
        else:
            tgtEmbeds=tgtEmbeds.view(1,batchSize,-1)

        out,hidden=self.decoder(tgtEmbeds,hidden)
        if not getAtt:
            return out,hidden,c_t
        else:
            return out,hidden,c_t,alphasNumpy

class DecoderRNN(nn.Module):
    def __init__(self,wids,vocabSize,emb_size,hidden_size):
        super(DecoderRNN,self).__init__()
        self.vocabSize=vocabSize
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.wids=wids

        
        self.embeddings=nn.Embedding(self.vocabSize,self.emb_size)
        self.decoder=nn.LSTM(self.emb_size,self.hidden_size)

    def forward(self,batchSize,tgtEmbedIndex,hidden,inference=False):
        tgtEmbeds=self.embeddings(tgtEmbedIndex)
        tgtEmbeds=tgtEmbeds.view(1,batchSize,-1)
        out,hidden=self.decoder(tgtEmbeds,hidden)
        return out,hidden





class LinearLayer(nn.Module):
    def __init__(self,inputSize,outputSize):
        super(LinearLayer,self).__init__()
        self.W=nn.Linear(inputSize,outputSize)

    def forward(self,inp):
        return self.W(inp)


###################################
class MultiRep(nn.Module):
    def __init__(self, vocab_sizes, embedding_sizes):
        super(MultiRep,self).__init__()
        self.all_embeddings = []
        for vocab_size,embedding_size in zip(vocab_sizes,embedding_sizes):
            embeddings=nn.Embedding(vocab_size, embedding_size)
            self.all_embeddings.append( embeddings )

    def _getEmbeddings(self, inp, i):
        embeds=self.all_embeddings[i](inp)
        return embeds
        # inp: batchSize x elementSize.  embeds: batchSize x elementSize x embeddingSize

    def forward(self,inp):
        all_embeds = []
        for i,inp_i in enumerate(inp):
            all_embeds.append(self._getEmbeddings(inp_i, i))
        all_concat_embeds=torch.cat(all_embeds, 2) # batchSize x elementSize x SUM_i(embeddingSize_i)
        return all_concat_embeds



class EncoderRNNMultiRep(nn.Module):
    def __init__(self, wids, vocab_sizes, emb_sizes, hidden_size, use_LSTM, share_embeddings, reference_embeddings=None):
        super(EncoderRNN,self).__init__()
        self.vocab_sizes=vocab_sizes
        self.emb_sizes=emb_sizes
        self.hidden_size=hidden_size
        self.use_LSTM=use_LSTM
        self.wids=wids

        if share_embeddings:
            self.embeddings=reference_embeddings #Here reference embeddings must be MultiRep obj
        else:
            self.embeddings=MultiRep(self.vocab_sizes,self.emb_sizes) #nn.Embedding(self.vocabSize,self.emb_size)

        if self.use_LSTM:
            self.encoder=nn.LSTM(self.emb_size,self.hidden_size)
        else:
            self.encoder=nn.GRU(self.emb_size,self.hidden_size)


    def forward(self,batchSize, embed_indices, hidden):
        embeds=self.embeddings(embed_indices) # embed_indices: LIST of batch_size x elements_size
        out,hidden=self.encoder(embeds.view(1,batchSize,-1),hidden) # ret is batch_size x element_size x embedding_size
        return out,hidden



################################################


class AttnDecoderRNNMutipleInput(nn.Module):
    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,use_attention,share_embeddings,reference_embeddings=None,extra_dims=0,typ="entire_as_tuple"):
        super(AttnDecoderRNNMutipleInput,self).__init__()
        self.vocabSize=vocabSize
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.use_LSTM=use_LSTM
        self.use_attention=use_attention
        self.wids=wids
        self.typ=typ

        if share_embeddings:
            self.embeddings=reference_embeddings
        else:
            self.embeddings=nn.Embedding(self.vocabSize,self.emb_size)

        if self.use_attention:
            if self.use_LSTM:
                self.decoder=nn.LSTM(self.emb_size+extra_dims,self.hidden_size)
            else:
                self.decoder=nn.GRU(self.emb_size+extra_dims,self.hidden_size)
        else:
            if self.use_LSTM:
                self.decoder=nn.LSTM(self.emb_size,self.hidden_size)
            else:
                self.decoder=nn.GRU(self.emb_size,self.hidden_size)


    def combineEncoderOutputs(self, encoder_outputs_list):
        if self.typ=="two_tuple" or self.typ=="three_tuple":
            return torch.add(encoder_outputs_list[0],encoder_outputs_list[1])

    def _computeAttention(self, encoderOutTensor, query):
        #prev_board_encoder_out = lst_of_encoderOutTensor[0] # board_len x outsize
        #cur_boad_encoder_out = lst_of_encoderOutTensor[1] # board_len x outsize
        #move_rep = lst_of_encoderOutTensor[2] # outsize
        #encoderOutTensor = prev_board_encoder_out + cur_boad_encoder_out

        o_t_expanded=query.expand(encoderOutTensor.size())
        dotProduct=torch.transpose(torch.sum(torch.mul(encoderOutTensor,o_t_expanded),2),0,1)
        alphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
        c_t=torch.squeeze(torch.sum(alphas*encoderOutTensor,0),0)
        return c_t


    def forward(self, batchSize, tgtEmbedIndex, lst_of_encoderOutTensor, o_t, hidden, feedContextVector=False, contextVector=None, inference=False):

        if self.use_attention:
            if not feedContextVector:
                c_t=self._computeAttention(lst_of_encoderOutTensor,o_t)
                del o_t
            else:
                c_t=contextVector

        tgtEmbeds=self.embeddings(tgtEmbedIndex)
        if inference:
            c_t=c_t.view(1,-1)
        if self.use_attention:
            tgtEmbeds=torch.cat([tgtEmbeds,c_t],1).view(1,batchSize,-1)
        else:
            tgtEmbeds=tgtEmbeds.view(1,batchSize,-1)

        out,hidden=self.decoder(tgtEmbeds,hidden)
        return out,hidden,c_t



class AttnDecoderRNNMutipleInputCNN(nn.Module):
    def __init__(self,wids,vocabSize,emb_size,hidden_size,use_LSTM,use_attention,share_embeddings,reference_embeddings=None,extra_dims=0,typ="entire_as_tuple",cnn_output_size=-1):
        super(AttnDecoderRNNMutipleInputCNN,self).__init__()
        self.vocabSize=vocabSize
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.use_LSTM=use_LSTM
        self.use_attention=use_attention
        self.wids=wids
        self.typ=typ
        self.cnn_output_size = cnn_output_size
        self.extra_dims = extra_dims

        if share_embeddings:
            self.embeddings=reference_embeddings
        else:
            self.embeddings=nn.Embedding(self.vocabSize,self.emb_size)

        if self.use_attention:
            extra_dims+=(2*cnn_output_size)
            self.decoder=nn.LSTM(self.emb_size+extra_dims,self.hidden_size)
        else:
            self.decoder=nn.LSTM(self.emb_size,self.hidden_size)

        self.fc = nn.Linear( hidden_size, cnn_output_size)
        self.fc_move = nn.Linear( hidden_size, self.extra_dims)
	#self.query

    def _computeAttention(self, lst_of_encoderOutTensor, query):

        encoderOutTensor = lst_of_encoderOutTensor[2]
        #print encoderOutTensor.data.shape, query.data.shape
	query_move = self.fc_move(query)
        o_t_expanded=query_move.expand(encoderOutTensor.size())
        #print "o_t_expanded : ",o_t_expanded.data.shape
        dotProduct=torch.transpose(torch.sum(torch.mul(encoderOutTensor,o_t_expanded),2),0,1)
        alphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
        c_t=torch.squeeze(torch.sum(alphas*encoderOutTensor,0),0)

        encoderOutTensor = lst_of_encoderOutTensor[0]
        query_cnn = self.fc(query)
        #print encoderOutTensor.data.shape, query_cnn.data.shape
        o_t_expanded=query_cnn.expand(encoderOutTensor.size())
        #print "o_t_expanded : ",o_t_expanded.data.shape
        dotProduct=torch.transpose(torch.sum(torch.mul(encoderOutTensor,o_t_expanded),2),0,1)
        alphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
        c_t_cnn=torch.squeeze(torch.sum(alphas*encoderOutTensor,0),0)

        encoderOutTensor = lst_of_encoderOutTensor[1]
        query_cnn = self.fc(query)
        o_t_expanded=query_cnn.expand(encoderOutTensor.size())
        dotProduct=torch.transpose(torch.sum(torch.mul(encoderOutTensor,o_t_expanded),2),0,1)
        alphas=torch.transpose(F.softmax(dotProduct),0,1).unsqueeze(2).expand(encoderOutTensor.size())
        c_t_cnn2=torch.squeeze(torch.sum(alphas*encoderOutTensor,0),0)

        c_t = torch.cat( [c_t,c_t_cnn,c_t_cnn2], 1 )

        return c_t


    def forward(self, batchSize, tgtEmbedIndex, lst_of_encoderOutTensor, o_t, hidden, feedContextVector=False, contextVector=None, inference=False):

        #print "hidden : ",hidden

        if self.use_attention:
            if not feedContextVector:
                c_t=self._computeAttention(lst_of_encoderOutTensor,o_t)
                del o_t
            else:
                c_t=contextVector

        tgtEmbeds=self.embeddings(tgtEmbedIndex)
        if inference:
            c_t=c_t.view(1,-1)
        if self.use_attention:
            #print c_t.data.shape
            tgtEmbeds=torch.cat([tgtEmbeds,c_t],1).view(1,batchSize,-1)
        else:
            tgtEmbeds=tgtEmbeds.view(1,batchSize,-1)

        out,hidden=self.decoder(tgtEmbeds,hidden)
        return out,hidden,c_t



class CNN(nn.Module):
    def __init__(self, inp_channels=1, output_channels=16, kernel_size=2, padding=1):
        super(CNN, self).__init__()
        self.output_channels = output_channels
        self.layer1 = nn.Conv2d(inp_channels, output_channels/2, kernel_size=kernel_size, padding=padding) # input channels, output channels, kernel_size, padding
        self.layer2 = nn.Sequential(
            nn.BatchNorm2d(output_channels/2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(output_channels/2, output_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(2))
        '''self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        '''
        #self.fc = nn.Linear(4*4*output_channels, 10)

    def forward(self, x):
        out = self.layer1(x)
        #print out.data.shape
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.layer2(out)
        #print out.data.shape
        out = out.view(out.size(0), self.output_channels, -1)
        #print out.data.shape
        #out = self.fc(out)
        return out
