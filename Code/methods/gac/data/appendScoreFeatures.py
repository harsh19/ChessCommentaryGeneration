import sys
import pickle
import math

def takeLog(x):
    sign=None
    if x<0:
        sign="<neg>"
        x=-x
    elif x==0:
        sign="<zero>"
    elif math.isnan(x):
        sign="<nan>"
    else:
        sign="<pos>"

    magString=None
    if sign=="<zero>" or x==0.0:
        magString="<NegInf>"
    elif sign=="<nan>":
        magString="<nan>"
    else:
        magString="<"+str(int(math.log(x)))+">"

    return sign+" "+magString

    

split=sys.argv[1]
moveClass=sys.argv[2]

all_feats=pickle.load(open(split+".che-eng."+moveClass+".score.all_feats.pickle","rb"))

transFeatStrings=[]
for feat in all_feats:
    transFeats=[takeLog(x) for x in feat]
    transFeatString=" ".join(transFeats)
    transFeatStrings.append(transFeatString)

inSrcLines=open(split+".che-eng."+moveClass+"attack.che").readlines()
outSrcFile=open(split+".che-eng."+moveClass+"score.che","w")

for transFeatString,inSrcLine in zip(transFeatStrings,inSrcLines):
    outLine=transFeatString+" "+inSrcLine
    outSrcFile.write(outLine)

outSrcFile.close()
