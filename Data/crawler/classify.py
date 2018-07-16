import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import json
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np

num_classes=5



def pre(use_annotated_data=True, use_rules_data=False): #use_annotated_data=True, oversampling_factor=3):
    
    all_data = []
    all_labels = [[]]
    for j in range(num_classes): all_data.append([])
    for j in range(num_classes): all_labels.append([])

    if use_rules_data:
        f = sys.argv[1]
        data = open(f,"r").readlines()
        data = [row.strip().split('||||') for row in data]
        for row in data:
            words = row[0].replace('[','').replace(']','').replace("'","")
            words = words.split(',')
            labels = json.loads(row[1])
            if labels[-1]>=0.99: #ignore data points of others class
            	continue
           	labels = labels[0:-1]
            for j,label in enumerate(labels):
                if label>0:
                    label=1
                else:
                    label=0
                all_data[j].append(words)
                all_labels[j].append(label) #[label,1-label])

    all_data_ann = []
    for j in range(num_classes): all_data_ann.append([])
    all_labels_ann = [[]]
    for j in range(num_classes): all_labels_ann.append([])

    if use_annotated_data:
        f = sys.argv[2]
        data = open(f,"r").readlines()
        print "Headers are as follows: "
        for dd in data[0].split('\t'):
            print "--> ",dd
        print "---------------------------"
        data = data[1:]
        data = [row.strip().split('\t') for row in data]
        for i,row in enumerate(data):
            #print row[1:],i
            words = row[0].replace('[','').replace(']','').replace("'","")
            words = words.split()
            labels = [int(p) for p in row[1:]]
            labels = labels #[0:-2] # ignore llast 2 labels
            for j,label in enumerate(labels):
                if label>0:
                    label=1
                else:
                    label=0
                all_data_ann[j].append(words)
                all_labels_ann[j].append(label) #[label,1-label])

    return all_data, all_labels, all_data_ann, all_labels_ann

#==========================================================================
def extractFeats(data):
    dct={}
    ctr=0
    counters={}
    for row in data:
        for word in data:
            if word not in dct:
                dct[word]=ctr
                counters[word]=1
                ctr+=1
            else:
                counters[word]+=1
    thresh=5
    dct_updated = {}
    valid_items = sorted(counters.items(), key=lambda x:-x[1] )
    for item in valid_items:
        word,cnt = item


def extractTFidfFeats(data,min_df=2): # tfidf feats
    data = [' '.join(row) for row in data]
    count_vect = CountVectorizer(min_df=min_df)
    X_train_counts = count_vect.fit_transform(data)
    tf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print "train shape : ", X_train_tf.shape
    #print X_train_tf
    return X_train_tf, tf_transformer,count_vect

def getTestFeats(data,count_vect,tf_transformer):
    data = [' '.join(row) for row in data]
    X_train_counts = count_vect.transform(data)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print "test shape : ", X_train_tf.shape
    return X_train_tf

#==========================================================================

def getAcc(pred, gt):
    #pred = np.argmax(preds_probs, axis=1)
    acc = sum(pred==gt)/(1.0*len(pred))
    print "acc = ",acc
    cm = [ [0,0],[0,0] ]
    for i in range(len(pred)):
        cm[gt[i]][pred[i]]+=1
    print "cm = ",cm

def classifier(data, labels):
    spl = 100
    data_train = data[:-spl]
    labels_train = labels[:-spl]
    labels_test = labels[-spl:]
    data_test = data[-spl:]
    X_train_tf, tf_transformer,count_vect = extractTFidfFeats(data_train)
    X_train_tf = X_train_tf.toarray()
    ## X_test_tf = getTestFeats(data_test, count_vect, tf_transformer)
    ## X_test_tf = X_test_tf.toarray()
    clf = LogisticRegression().fit(X_train_tf, np.array(labels_train))
    #probs = clf.predict_proba(X_test_tf)
    pred = predictor(clf, data_test, count_vect, tf_transformer)
    getAcc(pred,labels_test)
    return clf, tf_transformer,count_vect


def predictor(clf, unlabelled_data, count_vect, tf_transformer):
    X_test_tf = getTestFeats(unlabelled_data, count_vect, tf_transformer)
    X_test_tf = X_test_tf.toarray()
    print " x test shape : ", X_test_tf.shape
    probs = clf.predict_proba(X_test_tf)
    thresh= 0.50 #0.35
    #print probs[:100]
    preds = np.array( [ 0 if pr[1]<thresh else 1 for pr in probs ] )
    return preds

#==========================================================================

def main():
    all_data, all_labels, data_ann, label_ann = pre()
    for j in range(num_classes): print len(all_data[j]), len(data_ann[j])

    for j in range(num_classes): all_data[j].extend(data_ann[j])
    ##for j in range(num_classes): all_data[j].extend(data_ann[j])
    for j in range(num_classes): all_labels[j].extend(label_ann[j])
    ##for j in range(num_classes): all_labels[j].extend(label_ann[j])

    for j in range(num_classes): print len(all_data[j]), len(data_ann[j])

    class_num=0
    for j,(data,labels) in enumerate(zip(all_data,all_labels)):
        print " ==================================================\n class = ",j, len(data)
        #if sum(labels)<2:
        #   continue
        clf, tf_transformer, count_vect = classifier(data,labels)
        for f in sys.argv[3:]:
                f_unl = f #"saved_files/train.che-eng.single.en"
                data_unlabelled = open(f_unl,"r").readlines()
                data_unlabelled = [row.strip().split() for row in data_unlabelled]
                preds = predictor(clf, data_unlabelled, count_vect, tf_transformer)
                print " ",sum(preds)," are 1 out of ",len(preds)
        	for snt,pred in zip(data_unlabelled[:5],preds[:5]):
    	        	print snt, " --> " ,pred
                f = f + ".pred_labels_" + str(class_num)
                print "DUMPING predictions at ",f
                fw = open(f,"w")
                for pred in preds:
                	fw.write(str(pred) + "\n" )
              	fw.close()
              	print "----------------------"
        #break
	class_num+=1

main()

