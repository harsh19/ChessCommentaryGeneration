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
#import gc
#import sys
import torch
from torch.autograd import Variable
from models import *
from beam_search import *



class CLM(nn.Module):
    
    def __init__(self, params, start_symbol, end_symbol):
        super(CLM,self).__init__()
        self.decoder = AttnDecoderRNN(vocab_size=params.vocab_size, emb_size=params.decoder_emb_size, hidden_size=params.decoder_hidden_size, encoder_feat_size=params.encoder_feat_size)
        self.decoder_init_tranformer = nn.Linear(params.encoder_feat_size, params.decoder_hidden_size)
        self.W = nn.Linear(params.decoder_hidden_size, params.vocab_size) # use context vector also
        self.start_symbol = start_symbol
        self.end_symbol = end_symbol
        print "SimpleModel: ",self._modules.keys()
        for param in self.parameters():
            print(type(param.data), param.size())
        print "="*10

    def _getLongVariable(self,row,inference=False):
        tensor=torch.LongTensor(row)
        if torch.cuda.is_available():
            tensor=tensor.cuda()
        return autograd.Variable(tensor,volatile=inference)

    def _getInitialDecoderState(self, encoder_outputs):
        # encoder_outputs: B*49*512
        transformed_encoder_states = self.decoder_init_tranformer(encoder_outputs) # B*49*hidden_size
        decoder_init = transformed_encoder_states
        #print "decoder_init = ", decoder_init
        #torch.mean(transformed_encoder_states,1) # B,hidden_size
        return (decoder_init,decoder_init)

    def _beamDecode(self, encoder_outputs, max_len_decode, previous_output, previous_hidden):

            length_normalization_factor = 0.0
            beam_size = 5
            hidden = previous_hidden
            output = previous_output
            start_symbol = self.start_symbol
            end_symbol = self.end_symbol

            initial_beam = Caption(
                sentence=[self.start_symbol], #argmax instead ?
                state=(hidden[0], hidden[1]),
                logprob=0.0,
                score=0.0,
                metadata=[""])
            partial_captions = TopN(beam_size)
            partial_captions.push(initial_beam)
            complete_captions = TopN(beam_size)

            for _ in range(max_len_decode): #self.cnfg.TGT_LEN_LIMIT:
                encoder_outs_combined_beam = []
                row_beam = []
                o_t_beam = []
                hidden_beam0 = []
                hidden_beam1 = []
                partial_captions_list = partial_captions.extract()
                partial_captions.reset()

                for partial_caption in partial_captions_list:
                    encoder_outs_combined_beam.append(encoder_outputs)
                    hidden_beam0.append( partial_caption.state[0] )
                    hidden_beam1.append( partial_caption.state[1] )
                    row_beam.append(partial_caption.sentence[-1])
                    o_t_beam.append(partial_caption.state[1]) # should this be 0 or 1 ?

                row_beam =  self._getLongVariable(row_beam) #np.array(row_beam) #self.getIndex(row, inference=True) #np.array(row)
                encoder_outs_combined_beam = torch.stack(encoder_outs_combined_beam).squeeze(0)
                o_t_beam = torch.stack(o_t_beam).squeeze(0)
                hidden_beam0 = torch.stack(hidden_beam0).squeeze(0)
                hidden_beam0 = hidden_beam0.view(1,len(partial_captions_list),-1)
                hidden_beam1 = torch.stack(hidden_beam1).squeeze(0)
                hidden_beam1 = hidden_beam1.view(1,len(partial_captions_list),-1)
                hidden_beam = (hidden_beam0, hidden_beam1)
                #print "row_beam.shape[0] : ",row_beam.data.shape[0]

                output = o_t_beam
                hidden = hidden_beam0, hidden_beam1
                output, hidden = self._decoderStep(row_beam, encoder_outs_combined_beam, output, hidden)
                hidden = hidden[0].squeeze(0), hidden[1].squeeze(0)
                #print "---- ", hidden[0].data.shape, output.data.shape

                scores = F.log_softmax(self.W(output))
                scores = scores.cpu().data.numpy()

                for i, partial_caption in enumerate(partial_captions_list):
                    word_probabilities = scores[i]
                    #print "word_probabilities.shape : ",word_probabilities.shape
                    #print "hidden.shape : ",hidden[0].data.shape
                    state = (hidden[0][i], hidden[1][i])
                    # For this partial caption, get the beam_size most probable next words.
                    idxs = np.argpartition(word_probabilities, -beam_size)[-beam_size:]
                    # Each next word gives a new partial caption.
                    for w in idxs:
                        p = word_probabilities[w]
                        #if p < 1e-12:
                        #    continue  # Avoid log(0).
                        sentence = partial_caption.sentence + [w]
                        logprob = partial_caption.logprob + p #np.log(p)
                        #alphas_list = partial_caption.alphas + [context_vectors[i]]
                        score = logprob
                        if w == end_symbol:
                            if length_normalization_factor > 0:
                                score /= len(sentence)**length_normalization_factor
                            beam = Caption(sentence, state, logprob, score, metadata=[""])
                            complete_captions.push(beam)
                        else:
                            beam = Caption(sentence, state, logprob, score, metadata=[""])
                            partial_captions.push(beam)
                if partial_captions.size() == 0:
                    # We have run out of partial candidates; happens when beam_size = 1.
                    break

            complete_captions_list = complete_captions.extract()
            outputs = [ complete_caption.sentence for complete_caption in complete_captions_list ]
            return outputs, None

    def _greedyDecode(self, encoder_outputs, max_len_decode, previous_output, previous_hidden ):

        start_symbol = self.start_symbol
        end_symbol = self.end_symbol
        batch_size = encoder_outputs.data.shape[0]
        max_len = max_len_decode
        current_input = np.array([start_symbol]*batch_size)
        #print "current_input = ", current_input
        current_input = self._getLongVariable(current_input)
        outputs = []
        output_dist = []
        for j in range(batch_size):
            outputs.append([])
        done = np.zeros(batch_size)

        for t in range(max_len):
            #print "current_input = ", current_input
            #current_input = self._getLongVariable(current_input)
            previous_output, previous_hidden = self._decoderStep(current_input, encoder_outputs, previous_output, previous_hidden)
            scores = F.log_softmax(self.W(previous_output))
            output_dist.append(scores)
            maxValues,argmaxes = torch.max(scores,1)
            argmaxValue = argmaxes.cpu().data.numpy()
            current_input = []
            for j,val in enumerate(argmaxValue):
                if val == end_symbol:
                    done[j]=1
                if done[j]==0:
                    outputs[j].append(val)
                current_input.append(val)
            #print "* current_input = ", current_input
            current_input = self._getLongVariable(current_input)
            #print "** current_input = ", current_input


        return outputs, output_dist



    #def forward(self, image_feats, gt_output=None, mode="train", loss_function=None, get_loss=True 
    def forward(self, encoder_feats, gt_output=None, mode="train", loss_function=None, \
            get_loss=True ,max_len_decode=None, decoding_type="greedy"):

        start_symbol = self.start_symbol
        end_symbol = self.end_symbol

        # encoder_feats 
        encoder_outputs = encoder_feats #torch.stack(image_feats, dim=0)
        # encoderOutTensor = torch.stack([encoderOut for encoderOut in encoder_outs_j],dim=0)
        decoder_inital_state = self._getInitialDecoderState(encoder_outputs)
        decoder_initial_output = decoder_inital_state[0]
        previous_output, previous_hidden = decoder_initial_output, decoder_inital_state

        if mode=="train":

            # gt_output: B,len
            loss = 0.0
            gt_outputs_list_tensor = []
            batch_size = gt_output.shape[0]
            max_len = gt_output.shape[1]
            current_input = np.array([start_symbol]*batch_size)
            current_input = self._getLongVariable(current_input)
            decoderOuts = []

            for t in range(max_len):
                previous_output, previous_hidden = self._decoderStep(current_input, encoder_outputs, previous_output, previous_hidden)
                decoderOuts.append(previous_output) #.squeeze(0))
                current_input = self._getLongVariable(gt_output[:,t])
                gt_outputs_list_tensor.append(current_input)

            outputs = [ F.log_softmax(self.W(decoderOut)) for decoderOut in decoderOuts ]
            cross_entropy_loss = sum([loss_function(output,tgt) for output,tgt in zip(outputs,gt_outputs_list_tensor)])
            loss = cross_entropy_loss
            return loss

        elif mode=="decode":

            if decoding_type=="greedy":

                # gt_output: B,len
                outputs, output_dist = self._greedyDecode(encoder_outputs, max_len_decode, previous_output, previous_hidden )
                if get_loss:
                    gt_outputs_list_tensor = [ self._getLongVariable(val) for val in gt_output ]
                    cross_entropy_loss = sum([loss_function(output,tgt) for output,tgt in zip(output_dist,gt_outputs_list_tensor)]) # what if target length is greater than output #TODO
                    loss = cross_entropy_loss
                    return loss, outputs
                else:
                    return outputs

            elif decoding_type=="beam":

                # batch size is 1
                outputs, _ = self._beamDecode(encoder_outputs, max_len_decode, previous_output, previous_hidden )
                return outputs[0:1]

        else:
            print "---- Invalid mode----"

    def _decoderStep(self, current_input, encoder_outputs, previous_output, previous_hidden):
        batch_size = encoder_outputs.data.shape[0]
        #print "_decoderStep : ", current_input.data.shape, previous_output.data.shape, previous_hidden[0].data.shape
        out, hidden, context_vector, attn_weights = self.decoder( batch_size, current_input, encoder_outputs, previous_output, previous_hidden )
        return out,hidden
