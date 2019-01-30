import sys

inFile=open(sys.argv[1])
outFile=open(sys.argv[1]+".lower","w")


for line in inFile:
    outFile.write(line.lower())
