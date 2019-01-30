import sys
from convertToFEN import convertToFEN
from boardInfo import BoardInfo


split=sys.argv[1]

inSrcFile=open(split+".che-eng.2.che")
outSrcFile=open(split+".che-eng.2attack.che","w")

inTgtFile=open(split+".che-eng.2.en")
outTgtFile=open(split+".che-eng.2attack.en","w")

for line in inTgtFile:
	outTgtFile.write(line)

boardInfoObj=BoardInfo()
for line in inSrcFile:
    words=line.split()
    print words
    #outSrcFile.write(line[index+6:])
    #print words
    EOPIndex=words.index("<EOP>")
    EOCIndex=words.index("<EOC>")
    currentBoard=words[:EOCIndex]
    previousBoard=words[EOCIndex+1:EOPIndex]
    playerToMove=words[EOPIndex+1]
    playerToMoveNext=boardInfoObj.complement(playerToMove)
    #print playerToMove
    squareToMove=words[EOPIndex+4]
    #print squareToMove
    squareToMoveTo=words[EOPIndex+5]
    #print squareToMoveTo
    moveColor=None
    moveColorNext=None
    if playerToMove=="black":
        moveColor="b"
        moveColorNext="w"
    else:
        moveColor="w"
        moveColorNext="b"
    #print currentBoard
    #print previousBoard
    #print currentBoard.index("black_knight")
    #print previousBoard.index("black_knight")
    currentBoardFEN=convertToFEN(currentBoard,moveColorNext)
    previousBoardFEN=convertToFEN(previousBoard,moveColor)
    #print currentBoardFEN
    #print previousBoardFEN
    if squareToMove=="i3":
        squareToMove="f8"
    previousAttackerIndices=boardInfoObj.getAttackers(previousBoardFEN,boardInfoObj.complement(playerToMove),squareToMove)
    
    currentAttackerIndices=boardInfoObj.getAttackers(currentBoardFEN,playerToMove,squareToMoveTo,attacks=True)

    #print previousAttackerIndices
    previousAttackers=[]
    for previousAttackerIndex in previousAttackerIndices:
        previousAttackers.append(previousBoard[previousAttackerIndex].split("_")[0])
        previousAttackers.append(previousBoard[previousAttackerIndex].split("_")[1])
    previousAttackers.append("<EOPA>")
    previousAttackerString=" ".join(previousAttackers)

    currentAttackers=[]
    for currentAttackerIndex in currentAttackerIndices:
        #print currentBoard[currentAttackerIndex]
        if currentBoard[currentAttackerIndex]!="eps":
            color=currentBoard[currentAttackerIndex].split("_")[0]
            if color==playerToMoveNext:    
                currentAttackers.append(currentBoard[currentAttackerIndex].split("_")[0])
                currentAttackers.append(currentBoard[currentAttackerIndex].split("_")[1])
    currentAttackers.append("<EOCA>")
    currentAttackerString=" ".join(currentAttackers)



    #print previousAttackerString
    lineToPrint=" ".join(words[EOPIndex+1:])+" "+previousAttackerString+" "+currentAttackerString+"\n"
    outSrcFile.write(lineToPrint)
    print lineToPrint    
    #exit()

outTgtFile.close()
outSrcFile.close()
