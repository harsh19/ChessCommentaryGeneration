## GAC-CAT model

Training

```bash
$ python main.py -mode=<MODE> -modelName=<MODELNAME>  -problem=<PROBLEM> -MAX_TGT_SEQ_LEN=<MAXTARGET>
```
Example usage: 
```bash
$ python main.py -mode=train -modelName=cat_all_simple  -problem=CHESS7SIMPLE -MAX_TGT_SEQ_LEN=70 | tee ./logs/cat_all_simple.log
```

Run all_experiments.sh to run all the training and testing experiments.
