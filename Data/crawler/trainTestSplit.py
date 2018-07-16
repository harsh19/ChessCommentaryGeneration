import pickle
import sys
import math
from collections import defaultdict
import textUtils
import random

all_links=pickle.load(open("./saved_files/saved_links.p","r"))
extra_links=pickle.load(open("extra_pages.p","r"))

TRAIN_RATIO=0.7
VALID_RATIO=0.1
TEST_RATIO=1-TRAIN_RATIO-VALID_RATIO

all_links=[(i,all_links[i]) for i in range(len(all_links))]

random.seed(900659)
random.shuffle(all_links)

TRAIN_LENGTH=int(0.7*len(all_links))
VALID_LENGTH=int(0.1*len(all_links))

train_links=[all_links[i] for i in range(TRAIN_LENGTH)]
valid_links=[all_links[i] for i in range(TRAIN_LENGTH,VALID_LENGTH+TRAIN_LENGTH)]
test_links=[all_links[i] for i in range(VALID_LENGTH+TRAIN_LENGTH,len(all_links))]

print train_links[0]

pickle.dump(train_links,open("./saved_files/train_links.p","wb"))
pickle.dump(valid_links,open("./saved_files/valid_links.p","wb"))
pickle.dump(test_links,open("./saved_files/test_links.p","wb"))

