#import solver
import sys
import argparse


def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('-mode',dest='mode') #train,trial,inference Note: in train mode, supply only modelName, in inference mode, supply exact checkpoint path.
    parser.add_argument('-model_name',dest='model_name',default='default') #Name of the model; by default - 'default'
    parser.add_argument('-decoder_hidden_size',type=int,dest="decoder_hidden_size",help="dec_hidden_size",default=500)
    parser.add_argument('-max_train_sentences',type=int,dest="max_train_sentences",help="number of training sentences to use",default=99412) #How many training sentences? Default approx 1M (full for MT)
    parser.add_argument('-decoder_emb_size',type=int,dest="decoder_emb_size",help="emb_size",default=384)
    parser.add_argument('-encoder_feat_size',type=int,dest="encoder_feat_size",help="encoder_feat_size",default=1)
    parser.add_argument('-num_epochs',type=int,dest="num_epochs",help="num_epochs",default=11) #Generally about 10 epochs are sufficient. Pick model with best validation perplexity.
    parser.add_argument('-start',type=int,dest='start',help='start',default=1) #start token id
    parser.add_argument('-unk',type=int,dest='unk',help='unk',default=3) #unk token id
    parser.add_argument('-stop',type=int,dest='stop',help='stop',default=2) #stop token id
    parser.add_argument('-garbage',type=int,dest='garbage',help='garbage',default=0) #garbage token id (a.k.a PAD or pad)
    parser.add_argument('-min_tgt_frequency',type=int,dest='min_tgt_frequency',help='min_tgt_frequency',default=2) #Minimum frequency to not be UNKed on tgt side #TODO
    parser.add_argument('-max_tgt_seq_length',type=int,dest='max_tgt_seq_length',help='max_tgt_seq_length',default=50) #Maximum Target Sequence Length (set to high value to make unimportant)
    parser.add_argument('-max_decode_length',type=int,dest='max_decode_length',help='max_tgt_seq_length',default=20) #Maximum Target Sequence Length (set to high value to make unimportant)
    parser.add_argument('-print_step',type=int,dest='print_step',help='print_step',default=100) #Print every x minibatches
    parser.add_argument('-batch_size',type=int,dest='batch_size',help='batch_size',default=32) #Batch Size
    parser.add_argument('--debug', dest='debug', help='debug mode', action='store_true', default=False)
    #parser.add_argument('-method_type', dest='method_type', help='method_type', default='normal')
    parser.add_argument('-optimizer_type',dest='optimizer_type',help='optimizer_type',default="ADAM") # Optimizer: ADAM or SGD
    parser.add_argument('-model_dir',dest='model_dir',help='model_dir',default="tmp/") #Directory to save checkpoints and prediction files into
    parser.add_argument('-lang',dest='lang',help='lang',default="en") 
    parser.add_argument('-data_dir',dest='data_dir',help='dir',default="../../data/") 
    parser.add_argument('-typ',dest='typ',help='typ',default="che-eng.0") 
    parser.add_argument('-decoding_type',dest='decoding_type',help='decoding_type',default="greedy") # greedy, beam
    parser.add_argument('-feats',dest='feats',help='feats',default="101")
    parser.add_argument('-feats_dir',dest='feats_dir',help='feats_dir',default="../../chess_features/feature_dumps/")


    args=parser.parse_args()
    return args

params=parseArguments()
print params

import solver
obj = solver.Solver(params)
obj.main()
