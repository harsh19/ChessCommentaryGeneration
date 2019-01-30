
###### Training
python main.py -mode=train -modelName=desc_simple  -problem=CHESS0SIMPLE -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_simple.log
python main.py -mode=train -modelName=desc_attack  -problem=CHESS0ATTACK -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_attack.log
python main.py -mode=train -modelName=desc_score  -problem=CHESS0SCORE -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_score.log

python main.py -mode=train -modelName=qual_simple  -problem=CHESS1SIMPLE -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_simple.log
python main.py -mode=train -modelName=qual_attack  -problem=CHESS1ATTACK -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_attack.log
python main.py -mode=train -modelName=qual_score  -problem=CHESS1SCORE -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_score.log

python main.py -mode=train -modelName=compar_simple  -problem=CHESS2COMPARITIVESIMPLE  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_simple.log #maxtrain is 99k by default. need to set it if more than 99k sentences needed
python main.py -mode=train -modelName=compar_attack  -problem=CHESS2COMPARITIVEATTACK  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_attack.log
python main.py -mode=train -modelName=compar_score  -problem=CHESS2COMPARITIVESCORE  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_score.log

python main.py -mode=train -modelName=all_simple  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_simple.log
python main.py -mode=train -modelName=all_attack  -problem=CHESS7ATTACK -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_attack.log
python main.py -mode=train -modelName=all_score  -problem=CHESS7SCORE -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_score.log

#larger all
python main.py -mode=train -modelName=all_simple_e256_h500  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 -emb_size=256 -hidden_size=500 | tee ./logs/all_simple_e256_h500.log


####### Decoding

python main.py -mode=inference -modelName=tmp/desc_simple_3.ckpt  -problem=CHESS0SIMPLE -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_simple_inference.log
python main.py -mode=inference -modelName=tmp/desc_attack_3.ckpt  -problem=CHESS0ATTACK -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_attack_inference.log
python main.py -mode=inference -modelName=tmp/desc_score_3.ckpt  -problem=CHESS0SCORE -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70 | tee ./logs/desc_score_inference.log

python main.py -mode=inference -modelName=tmp/qual_simple_7.ckpt  -problem=CHESS1SIMPLE -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_simple_inference.log
python main.py -mode=inference -modelName=tmp/qual_attack_8.ckpt  -problem=CHESS1ATTACK -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_attack_inference.log
python main.py -mode=inference -modelName=tmp/qual_score_10.ckpt  -problem=CHESS1SCORE -max_train_sentences=571 -MAX_TGT_SEQ_LEN=70 | tee ./logs/qual_score_inference.log

python main.py -mode=inference -modelName=tmp/compar_simple_3.ckpt  -problem=CHESS2COMPARITIVESIMPLE  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_simple_inference.log #maxtrain is 99k by default. need to set it if more than 99k sentences needed
python main.py -mode=inference -modelName=tmp/compar_attack_5.ckpt  -problem=CHESS2COMPARITIVEATTACK  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_attack_inference.log
python main.py -mode=inference -modelName=tmp/compar_score_5.ckpt  -problem=CHESS2COMPARITIVESCORE  -MAX_TGT_SEQ_LEN=70 | tee ./logs/compar_score_inference.log

python main.py -mode=inference -modelName=tmp/all_simple_2.ckpt  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_simple_inference.log
python main.py -mode=inference -modelName=tmp/all_attack_3.ckpt  -problem=CHESS7ATTACK -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_attack_inference.log
python main.py -mode=inference -modelName=tmp/all_score_3.ckpt  -problem=CHESS7SCORE -MAX_TGT_SEQ_LEN=70 | tee ./logs/all_score_inference.log


###### Combine test predictions
cat tmp/desc_simple_3.ckpt.test.output tmp/qual_simple_7.ckpt.test.output tmp/compar_simple_3.ckpt.test.output > tmp/comb_simple.test.output
cat tmp/desc_attack_3.ckpt.test.output tmp/qual_attack_8.ckpt.test.output tmp/compar_attack_5.ckpt.test.output > tmp/comb_attack.test.output
cat tmp/desc_score_3.ckpt.test.output tmp/qual_score_10.ckpt.test.output tmp/compar_score_5.ckpt.test.output > tmp/comb_score.test.output




##### Bleu evaluation
./multi-bleu.perl -lc data/test.che-eng.0simple.en  < tmp/desc_simple_3.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.0attack.en  < tmp/desc_attack_3.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.0score.en  < tmp/desc_score_3.ckpt.test.output

./multi-bleu.perl -lc data/test.che-eng.1simple.en  < tmp/qual_simple_7.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.1attack.en  < tmp/qual_attack_8.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.1score.en  < tmp/qual_score_10.ckpt.test.output

./multi-bleu.perl -lc data/test.che-eng.2.comparitivesimple.en  < tmp/compar_simple_3.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.2.comparitiveattack.en  < tmp/compar_attack_5.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.2.comparitivescore.en  < tmp/compar_score_5.ckpt.test.output

./multi-bleu.perl -lc data/test.che-eng.7simple.en  < tmp/all_simple_2.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.7attack.en  < tmp/all_attack_3.ckpt.test.output
./multi-bleu.perl -lc data/test.che-eng.7score.en  < tmp/all_score_3.ckpt.test.output

./multi-bleu.perl -lc data/test.che-eng.7simple.en  < tmp/comb_simple.test.output
./multi-bleu.perl -lc data/test.che-eng.7attack.en  < tmp/comb_attack.test.output
./multi-bleu.perl -lc data/test.che-eng.7score.en  < tmp/comb_score.test.output


#### Bleu2 evaluation
./BLEU2.perl -lc data/test.che-eng.0simple.en  < tmp/desc_simple_3.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.0attack.en  < tmp/desc_attack_3.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.0score.en  < tmp/desc_score_3.ckpt.test.output

./BLEU2.perl -lc data/test.che-eng.1simple.en  < tmp/qual_simple_7.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.1attack.en  < tmp/qual_attack_8.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.1score.en  < tmp/qual_score_10.ckpt.test.output

./BLEU2.perl -lc data/test.che-eng.2.comparitivesimple.en  < tmp/compar_simple_3.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.2.comparitiveattack.en  < tmp/compar_attack_5.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.2.comparitivescore.en  < tmp/compar_score_5.ckpt.test.output

./BLEU2.perl -lc data/test.che-eng.7simple.en  < tmp/all_simple_2.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.7attack.en  < tmp/all_attack_3.ckpt.test.output
./BLEU2.perl -lc data/test.che-eng.7score.en  < tmp/all_score_3.ckpt.test.output

./BLEU2.perl -lc data/test.che-eng.7simple.en  < tmp/comb_simple.test.output
./BLEU2.perl -lc data/test.che-eng.7attack.en  < tmp/comb_attack.test.output
./BLEU2.perl -lc data/test.che-eng.7score.en  < tmp/comb_score.test.output
