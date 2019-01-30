# pytorch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

# python general
import numpy as np
import random
import math



#######################################

class AttnDecoderRNN(nn.Module):

    def __init__(self, vocab_size=100, emb_size=32, hidden_size=64, encoder_feat_size=512):

        super(AttnDecoderRNN,self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.hidden_size=hidden_size
        self.embeddings=nn.Embedding(self.vocab_size,self.emb_size)
        inp_size = emb_size + encoder_feat_size
        self.decoder=nn.LSTM(inp_size, self.hidden_size)
        self.enc_transform = nn.Linear(encoder_feat_size, hidden_size)
        self.hid_transform = nn.Linear(hidden_size, hidden_size)
        print "AttnDecoderRNN: ",self._modules.keys()
        #for param in self.parameters():
        #    print(type(param.data), param.size())
        #print "="*10


    def _getAllZeros(self): # dummy. for debugging.
        hiddenElem=torch.zeros(1,self.hidden_size) # 1,b,hidden_size
        if torch.cuda.is_available():
            hiddenElem=hiddenElem.cuda()
        hiddenElem = autograd.Variable(hiddenElem)
        return hiddenElem

    def _computeAttention(self, encoder_outputs, previous_output):
        '''encoder_outputs_transformed =   self.enc_transform(encoder_outputs) # B*49*hidden_size
        query = self.hid_transform(previous_output) # query: B*hidden_size
        query = query.unsqueeze(1) #(encoder_outputs_transformed.size()) # query: B*1*hidden_size
        dotProduct = torch.sum(torch.mul(encoder_outputs_transformed,query),2) #B,49
        attn_weights = F.softmax(dotProduct).unsqueeze(2) # B,49,1
        context_vector = torch.sum(attn_weights*encoder_outputs_transformed,1) # B*hidden_size
        return context_vector, attn_weights
        '''
        return encoder_outputs, None
        # context_vector: B*encoder_feat_size
        # attn_weights: B*49


    def forward(self, batch_size, current_input, encoder_outputs, previous_output, previous_hidden):
        '''
        encoder_outputs: batch_size*11. encoder_feat_size=11
        current_input: batch_size * 1
        previous_output: batch_size * hidden_size
        previous_hidden: tuple. each is batch_size * hidden_size
        '''
        #encoder_outputs = encoder_outputs.view(batch_size,-1,11) # batch_size*7*7*512 -> batch_size*49*512

        current_input_embeddings = self.embeddings(current_input) # current_input_embeddings: B*embedding_size
        context_vector, attn_weights = self._computeAttention(encoder_outputs, previous_output)
        rnn_input = torch.cat([current_input_embeddings,context_vector],1).view(1,batch_size,-1)
        previous_hidden = previous_hidden[0].view(1,batch_size,-1), previous_hidden[1].view(1,batch_size,-1)
        out,hidden=self.decoder(rnn_input, previous_hidden)
        out = out.squeeze(0) # 1,B,hideen -> B,hidden
        return out, hidden, context_vector, attn_weights
