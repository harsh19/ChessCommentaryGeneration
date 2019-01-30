import sys

split=sys.argv[1]
featurization=sys.argv[2]
entities=bool(sys.argv[3])
names=sys.argv[4:]

suffix=".en"
if False:
    suffix=".en.entities"


outSrcFile=open(split+".che-eng.7"+featurization+".che","w")
outTgtFile=open(split+".che-eng.7"+featurization+suffix,"w")

for index,name in enumerate(names):
    inSrcFileName=open(split+".che-eng."+name+featurization+".che")
    inTgtFileName=open(split+".che-eng."+name+featurization+suffix)
    inSrcLines=inSrcFileName.readlines()
    inTgtLines=inTgtFileName.readlines()

    for srcLine,tgtLine in zip(inSrcLines,inTgtLines):
        outSrcFile.write("<class."+name+">"+" "+srcLine)
        outTgtFile.write(tgtLine)


outSrcFile.close()
outTgtFile.close()    

