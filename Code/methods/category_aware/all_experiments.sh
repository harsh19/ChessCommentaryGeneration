
###### Training

python main.py -mode=train -modelName=cat_all_simple  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_simple.log
python main.py -mode=train -modelName=cat_all_attack  -problem=CHESS7ATTACK -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_attack.log
python main.py -mode=train -modelName=cat_all_score  -problem=CHESS7SCORE -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_score.log

#larger model
python main.py -mode=train -modelName=cat_all_simple_e256_h500  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 -emb_size=256 -hidden_size=500 | tee ./logs/cat_all_simple_larger_e256_h500.log



####### Decoding

python main.py -mode=inference -modelName=tmp/cat_all_simple_2.ckpt  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_simple_inference.log
python main.py -mode=inference -modelName=tmp/all_attack_3.ckpt  -problem=CHESS7ATTACK -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_attack_inference.log
python main.py -mode=inference -modelName=tmp/all_score_3.ckpt  -problem=CHESS7SCORE -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_score_inference.log

# larger model
python main.py -mode=inference -emb_size=256 -hidden_size=500 -modelName=tmp/cat_all_simple_e256_h500_2.ckpt  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_simple_e256h500_inference.log



##### Bleu evaluation
./multi-bleu.perl -lc data/test.che-eng.7simple.en  < tmp/cat_all_simple_e256_h500_2.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.7simple.en  < tmp/cat_all_simple_2.ckpt.test.output

#./multi-bleu.perl -lc data/test.che-eng.7simple.en  < tmp/comb_simple.test.output
#./multi-bleu.perl -lc data/test.che-eng.7attack.en  < tmp/comb_attack.test.output
#./multi-bleu.perl -lc data/test.che-eng.7score.en  < tmp/comb_score.test.output



#### Bleu2 evaluation
./BLEU2.perl -lc data/test.che-eng.7simple.en  < tmp/cat_all_simple_e256_h500_2.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.7simple.en  < tmp/cat_all_simple_2.ckpt.test.output

#./BLEU2.perl -lc data/test.che-eng.7simple.en  < tmp/comb_simple.test.output
#./BLEU2.perl -lc data/test.che-eng.7attack.en  < tmp/comb_attack.test.output
#./BLEU2.perl -lc data/test.che-eng.7score.en  < tmp/comb_score.test.output


