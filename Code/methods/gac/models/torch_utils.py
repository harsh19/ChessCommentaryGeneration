import numpy as np

def splitBatches(train=[],batch_size=32,padSymbol="PAD",method="post"):

    batches=[]
    masks=[]

    instanceIndex=0
    while instanceIndex<len(train):
        batch=train[instanceIndex:min(instanceIndex+batch_size,len(train))]
        batchLength=max([len(x) for x in batch])
        mask=[maskSeq(x,batchLength,padSymbol,method=method) for x in batch]
        batch=[padSeq(x,batchLength,padSymbol,method=method) for x in batch]
        mask=np.array(mask)
        batch=np.array(batch)
        batches.append(batch)
        masks.append(mask)

        instanceIndex+=batch_size

    return batches,masks

def maskSeq(seq,desired_length,pad_symbol,method="pre"):
    seq_length=len(seq)
    mask=[1,]*desired_length
    if len(seq)<desired_length:
        if method=="post":
            mask=[1,]*seq_length+[0,]*(desired_length-seq_length)
        else:
            mask=[0,]*(desired_length-seq_length)+[1,]*seq_length

    return mask
   

def padSeq(seq,desired_length,pad_symbol,method="pre"):
    seq_length=len(seq)

    if len(seq)<desired_length:
        if method=="post":
            seq=seq+[pad_symbol,]*(desired_length-seq_length)
        else:
            seq=[pad_symbol,]*(desired_length-seq_length)+seq

    return seq


def reverseDict(wids):
    idws={}
    for key in wids:
        idws[wids[key]]=key
    return idws

