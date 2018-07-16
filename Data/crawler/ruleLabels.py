import random
import sys

print "8888------------" 

NO_OF_CLASSES=7  #Quality, Alternate,Expected,Long-Term,State,Describe,Others
QUALITY,ALTERNATE,EXPECTED,LONG_TERM,STATE,DESCRIBE,OTHERS=0,1,2,3,4,5,6


# Read in Rules and Multi-Rules. We need something special for single-word rules because a word can be a substring of another word often.

#Single-Word Rules
rules={}
rules[QUALITY]=["solid","excellent","blunder","error","fortunately","nice","unwise","good","bad","disaster","mistake","disastrous","blunderous","blunder","premature","reasonable","weak","strong"]
rules[QUALITY]+=["blundered","lame","terrible","horrible","flexible","pretty"]
rules[ALTERNATE]=["considering","better","inclined","possible","possibility","alternately","alternative","if","going"]
rules[EXPECTED]=["curious","conventionally","logical","surprise","surprising","typical","atypical","expected","unexpected","standard","odd","strange","unusual","forced","Out","inevitable"]
rules[LONG_TERM]=["hopeful","need","later","important","critical","advantage","decisive","going","preparing","prospects","inevitable","probably","offer","analyzing","otherwise"]
rules[STATE]=["feeling","believe","points","edge","pressure"]
rules[DESCRIBE]=["removes","retreats","blocks","Moving","takes","plays","protects","forks","repositions","brings","bring"]
rules[DESCRIBE]+=["moves","attacks","defends","captures","sacrifices","blocks","attacking","castles","defending","took","decides","standard","chose","chosen","prefer","prefers"]
rules[OTHERS]=[]
#Multi-word rules
multiRules={}
multiRules[QUALITY]=[]
multiRules[ALTERNATE]=["got to","more inclined"]
multiRules[EXPECTED]=[]
multiRules[LONG_TERM]=["as then","and if","got to","idea to"]
multiRules[STATE]=["worse position","good position","description of game","description about condition"]
multiRules[DESCRIBE]=["give up","gives up","gave up","Fischer Variation","Sicilian Opening","this is the"]
multiRules[OTHERS]=[]

#Read rules from file
for line in open("rules.txt"):
    words=line[:-1].split(",")
    if len(words[1].split())>1:
        multiRules[int(words[0])].append(words[1])
    else:
        rules[int(words[0])].append(words[1])

def assignPseudoLabel(sentence):
    #sentence is a sequence of words    
    sentenceString=" ".join(sentence)
    #sentenceString is a seq
    aspect=[0.0,]*NO_OF_CLASSES

    noFired=True

    for rule in rules.keys():
        for word in rules[rule]:
            if word in sentence or word.capitalize() in sentence:
                aspect[rule]=1.0
                noFired=False

    for rule in multiRules.keys():
        for word in multiRules[rule]:
            if word in sentenceString or word.capitalize() in sentenceString:
                aspect[rule]=1.0
                noFired=False

    if noFired:
        aspect[NO_OF_CLASSES-1]=1.0
    else:
        Z=sum(aspect)
        aspect=[x/Z for x in aspect]


    return aspect


if __name__=="__main__":
    random.seed(1096493)
    inFile=open(sys.argv[1])
    indexedLines=[(i,line[:-1]) for i,line in enumerate(inFile.readlines())]
    random.shuffle(indexedLines)
    validationLines=indexedLines[10000:]
    indexedLines=indexedLines[:10000]
    indexedLines=[(x,y.split()) for x,y in indexedLines]
    indexedLines=[(x,y,assignPseudoLabel(y)) for x,y in indexedLines]
    labelDistribution={}
    for indexedLine in indexedLines:
        for aspectIndex,aspectValue in enumerate(indexedLine[2]):
            if aspectValue>0.00001:
                if aspectIndex not in labelDistribution:
                    labelDistribution[aspectIndex]=0
                labelDistribution[aspectIndex]+=1
    print labelDistribution

    outFile=open(sys.argv[1]+".pseudoLabels","w")
    print "outfile = ",outFile

    for indexedLine in indexedLines:
        outFile.write(str(indexedLine[1])+" |||| "+str(indexedLine[2])+"\n")

    outFile.close()

