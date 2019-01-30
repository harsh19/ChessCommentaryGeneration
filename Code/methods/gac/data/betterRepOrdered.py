import sys
from convertToFEN import convertToFEN
from boardInfo import BoardInfo


split=sys.argv[1]
moveClass=sys.argv[2]
order=int(sys.argv[3])
featureType=sys.argv[4] #simple or attack

if split=="all": 
    splits=["train","valid","test"]
else:
    splits=[split,]

for split in splits:
    inSrcFile=open(split+".che-eng."+moveClass+".che")

    outSrcFile=open(split+".che-eng."+moveClass+featureType+str(order)+".che","w")

    inTgtFile=open(split+".che-eng."+moveClass+".en")

    outTgtFile=open(split+".che-eng."+moveClass+featureType+str(order)+".en","w")

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
        
        playerToMoveString=words[EOPIndex+1]+" "+words[EOPIndex+2]
        pieceToMoveString=words[EOPIndex+3]
        squareToMove=words[EOPIndex+4]
        squareToMoveTo=words[EOPIndex+5]
        restOfStuff=" ".join(words[EOPIndex+6:-1]+["<EORest>",])
        finalTag=words[-1]
        print playerToMoveString
        print pieceToMoveString
        print squareToMove
        print squareToMoveTo
        print restOfStuff
        print finalTag

        moveOrder=None

        if order==0: 
            moveOrder=[playerToMoveString,pieceToMoveString,squareToMove,squareToMoveTo,restOfStuff,finalTag]
        elif order==1:
            moveOrder=[squareToMoveTo,restOfStuff,pieceToMoveString,squareToMove,playerToMoveString,finalTag]
        elif order==2:
            moveOrder=[pieceToMoveString,squareToMove,playerToMoveString,squareToMoveTo,restOfStuff,finalTag]
        elif order==3:
            moveOrder=[restOfStuff,squareToMove,playerToMoveString,pieceToMoveString,squareToMoveTo,finalTag]
        elif order==4:
            moveOrder=[squareToMove,pieceToMoveString,squareToMoveTo,restOfStuff,playerToMoveString,finalTag]

        print len(moveOrder)
        moveString=" ".join(moveOrder)
        print "Move String:",moveString

        finalOrder=None
        if featureType=="attack":
            if order==0:
                finalOrder=[moveString,previousAttackerString,currentAttackerString]
            elif order==1:
                finalOrder=[previousAttackerString,moveString,currentAttackerString]
            elif order==2:
                finalOrder=[previousAttackerString,currentAttackerString,moveString]
            elif order==3:
                finalOrder=[currentAttackerString,moveString,previousAttackerString]
            elif order==4:
                finalOrder=[currentAttackerString,previousAttackerString,moveString]
        elif featureType=="simple":
            finalOrder=[moveString,]

        newLineToPrint=" ".join(finalOrder)+"\n"
        outSrcFile.write(newLineToPrint)
        print lineToPrint    
        print newLineToPrint

    outTgtFile.close()
    outSrcFile.close()
