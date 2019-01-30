import sys
import pickle

features=pickle.load(open(sys.argv[1],"rb"))

#print len(features)

testOutputs=[]

for point in features:
    print point[-2] #,point[-1]
    if point[-2]>-7.25:
        testOutputs.append("A good move .")
    else:
        testOutputs.append("A bad move .")


outFile=open(sys.argv[2],"w")
for testOutput in testOutputs:
    outFile.write(testOutput+"\n")

outFile.close()
