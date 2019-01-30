
## Generate Features

```bash
$ python generate_features.py argv1 argv2
```

- src_dir = sys.argv[1] # ./data/
- src = sys.argv[2] # e.g. "train.che-eng.0"
- This will create relevant feature dumps of all three types
- e.g. python generate_features.py ../data/ train.che-eng.2 > feature_dumps/log_train.2.all
- within code can comment out feature types to get outputs of only specific features

