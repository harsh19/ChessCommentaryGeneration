#import solver
import sys
import argparse
import copy

def parseArguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('-mode',dest='mode') #train,trial,inference,trainlm Note: in train mode, supply only modelName, in inference mode, supply exact checkpoint path.
    parser.add_argument('-modelName',dest='modelName',default='default') #Name of the model; by default - 'default'
    parser.add_argument('-lmName',dest='lmName',default='default') #Name of the model; by default - 'default'
    parser.add_argument('-hidden_size',type=int,dest="hidden_size",help="hidden_size",default=384) #Encoder and Decoder state size (assumed equal)
    parser.add_argument('-max_train_sentences',type=int,dest="max_train_sentences",help="number of training sentences to use",default=99412) #How many training sentences? Default approx 1M (full for MT)
    parser.add_argument('-emb_size',type=int,dest="emb_size",help="emb_size",default=192) #Embedding size, both encoder and decoder (assumed equal)
    parser.add_argument('-layer_depth',type=int,dest="layer_depth",help="layer_depth",default=1) #For the future, currently this is always 1 in the code.
    parser.add_argument('-NUM_EPOCHS',type=int,dest="NUM_EPOCHS",help="NUM_EPOCHS",default=15) #Generally about 10 epochs are sufficient. Pick model with best validation perplexity.
    parser.add_argument('-start',type=int,dest='start',help='start',default=0) #start token id
    parser.add_argument('-unk',type=int,dest='unk',help='unk',default=1) #unk token id
    parser.add_argument('-stop',type=int,dest='stop',help='stop',default=2) #stop token id
    parser.add_argument('-garbage',type=int,dest='garbage',help='garbage',default=3) #garbage token id (a.k.a PAD or pad)
    parser.add_argument('-min_src_frequency',type=int,dest='min_src_frequency',help='min_src_frequency',default=1) #Minimum frequency to not be UNKed on src side
    parser.add_argument('-min_tgt_frequency',type=int,dest='min_tgt_frequency',help='min_tgt_frequency',default=1) #Minimum frequency to not be UNKed on tgt side
    parser.add_argument('-MAX_SEQ_LEN',type=int,dest='MAX_SEQ_LEN',help='MAX_SEQ_LEN',default=500) #Maximum Length (set to high value to make unimportant)
    parser.add_argument('-MAX_TGT_SEQ_LEN',type=int,dest='MAX_TGT_SEQ_LEN',help='MAX_TGT_SEQ_LEN',default=90) #Maximum Target Sequence Length (set to high value to make unimportant)
    parser.add_argument('-problem',dest='problem',help='problem',default="CHESS") #"MT" or "CHESS" or "CHESS0" or "CHESSLABEL" or "CHESS0SIMPLE" or "CHESS0ATTACK" or "CHESS1SIMPLE" or "CHESS1ATTACK"
    parser.add_argument('-method',dest='method',help='method',default="greedy") #"greedy or "beam" or "beamLM" or "beamSib"
    parser.add_argument('-beamSize',type=int,dest='beamSize',help='beamSize',default=3) #
    parser.add_argument('-p',type=float,dest='p',help='p',default=0.3) #p-value
    parser.add_argument('-PRINT_STEP',type=int,dest='PRINT_STEP',help='PRINT_STEP',default=500) #Print every x minibatches
    parser.add_argument('-batch_size',type=int,dest='batch_size',help='batch_size',default=32) #Batch Size



    parser.add_argument('-optimizer_type',dest='optimizer_type',help='optimizer_type',default="ADAM") # Optimizer: ADAM or SGD
    parser.add_argument('-model_dir',dest='model_dir',help='model_dir',default="tmp/") #Directory to save checkpoints and prediction files into

    #Flags to turn default True arguments off. Read carefully.
    parser.add_argument('-no_use_reverse',action='store_false',dest='use_reverse',default=True) #Boolean switch - use only to turn reversing off
    parser.add_argument('-no_init_mixed',action='store_false',dest='init_mixed',default=True) #Init decoder state with avg of last forward and last backward - use only to turn false
    parser.add_argument('-no_use_attention',action='store_false',dest='use_attention',default=True) #Attention on-off switch - use only to turn false
    parser.add_argument('-no_use_LSTM',action='store_false',dest='use_LSTM',default=True) #LSTM on-off switch - use only to turn false - in that case GRUs are used
    parser.add_argument('-no_use_downstream',action='store_false',dest='use_downstream',default=True) #Whether to use context vector downstream - use only to turn false
    parser.add_argument('-no_srcMasking',action='store_false',dest='srcMasking',default=True) #Whether src side masking is on - use only to turn false
    parser.add_argument('-no_mem_optimize',action='store_false',dest='mem_optimize',default=True) #Memory optimizations (deleting local variables in advance etc) - use only to turn false

    #Flags to turn default False arguments off. Read carefully.
    parser.add_argument('-init_enc',action='store_true',dest='init_enc',default=False) #Use the forward encoder state for initializing decoder. If false the backward encoder (a.k.a revcoder) is used.
    parser.add_argument('-share_embeddings',action='store_true',dest='share_embeddings',default=False) #Share encoder and decoder embeddings
    parser.add_argument('-normalizeLoss',action='store_true',dest='normalizeLoss',default=False) #Whether to normalize loss per minibatch
    parser.add_argument('-decoder_prev_random',action='store_true',dest='decoder_prev_random',default=False) #Randomly use previous context vector with prob p at decoding time
    parser.add_argument('-context_dropout',action='store_true',dest='context_dropout',default=False)
    parser.add_argument('-mixed_decoding',action='store_true',dest='mixed_decoding',default=False)
    parser.add_argument('-cudnnBenchmark',action='store_true',dest='cudnnBenchmark',default=False) #CuDNN Benchmark (purported speedup)
    parser.add_argument('-initGlove',action='store_true',dest='initGlove',default=False) #Init target embeddings which overlap with glove with glove.
    parser.add_argument('-getAtt',action='store_true',dest='getAtt',default=False) #Whether to dump attentions in inference mode for all test examples.
    parser.add_argument('-sigmoid',action='store_true',dest='sigmoid',default=False) #Replaces softmax attention by sigmoid attention


    # Flags specific to solver_general
    parser.add_argument('--harsh',action='store_true',dest="use_general_solver",default=False)
    parser.add_argument('-typ',dest="typ",default="two_tuple",help="input format")
    parser.add_argument('-useLM',action='store_true',dest='useLM',default=False)

    #TGT_LEN_LIMIT=1000

    args=parser.parse_args()
    return args

params=parseArguments()
params.lmObj=None
print params

if params.use_general_solver:
    import solver_general as solver
    seq2seq=solver.Solver(params)
    print "TYP = ", params.typ
    seq2seq.main(typ=params.typ)
elif params.useLM:
    import solver_lm as solver
    seq2seq=solver.Solver(params)
    seq2seq.main()
else:
    lmObj=None
    if params.mode=="inference" and params.method=="beamLM":
        print "Loading LM"
        import solver_lm as solver_lm
        copiedParams=copy.deepcopy(params)
        copiedParams.mode="saveLM"
        copiedParams.modelName=params.lmName
        lmObj=solver_lm.Solver(copiedParams)
        lmObj.main()
        params.lmObj=lmObj
        print "Done Loading LM"
    import solver
    seq2seq=solver.Solver(params)
    seq2seq.main()
