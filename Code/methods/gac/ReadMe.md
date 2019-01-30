## GAC model

Training

```bash
$ python main.py -mode=<MODE> -modelName=<MODELNAME>  -problem=<PROBLEM> -max_train_sentences=<MAXTRAIN> -MAX_TGT_SEQ_LEN=<MAXTARGET>
```
Example usage: 
```bash
$ python main.py -mode=train -modelName=desc_simple  -problem=CHESS0SIMPLE -max_train_sentences=20089 -MAX_TGT_SEQ_LEN=70
```

Run all_experiments.sh to run all the training and testing experiments.
