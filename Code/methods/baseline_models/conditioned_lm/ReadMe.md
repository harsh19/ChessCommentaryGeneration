## GAC-sparse

Sanity check:
```bash
$ python main.py -mode=trial
```

Training
```bash
$ python main.py -mode=train -model_name=gacsparse
```

Testing
```bash
$ python main.py -mode=inference -model_name=tmp/gacsparse_4.ckpt
```
