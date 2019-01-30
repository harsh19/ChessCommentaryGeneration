import sys

gtFile=open(sys.argv[1])
stFile=open(sys.argv[2])
gtoFile=open(sys.argv[1]+".strip","w")
oFile=open(sys.argv[2]+".strip","w")


gtLines=gtFile.readlines()
stLines=stFile.readlines()

accuracy=0.0
total=len(gtLines)

for x,y in zip(gtLines,stLines):
    gtWords=x.split()
    stWords=y.split()

    if gtWords[0]==stWords[0]:
        accuracy+=1.0

    oFile.write(" ".join(stWords[1:])+"\n")
    gtoFile.write(" ".join(gtWords[1:])+"\n")


print "Accuracy:",accuracy/total

gtoFile.close()
oFile.close()
