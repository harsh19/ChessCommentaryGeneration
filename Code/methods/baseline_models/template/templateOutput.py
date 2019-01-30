import sys

def getTemplate(movingPlayer,pieceMoved,squareFrom,squareTo,capture=False,capturedPiece=None,check=False,castle=False,castlingType=None):
    response=None
    if not capture and not castle:
        response=movingPlayer.title()+" moves the "+pieceMoved+" from "+squareFrom+" to "+squareTo
    elif capture:
        response=movingPlayer.title()+" captures the "+capturedPiece+" at "+squareTo+" using the "+pieceMoved+" at "+squareFrom
    elif castle:
        response=movingPlayer.title()+" does a "+castlingType+" castling"

    if check:
        response=response+" ,  putting the king in check"
    
    return response

def getResponse(line):
    words=line.split()
    
    EOMIndex=words.index("<EOM>")

    movingPlayer=words[0]
    pieceMoved=words[2]
    squareFrom=words[3]
    squareTo=words[4]
    capture=None
    capturedPiece=None
    if "capture" in words:
        capture=True
        capturedPiece=words[EOMIndex-1]

    castle="castling" in words
    castlingType=None
    if castle:
        castlingType=words[EOMIndex-2]
    check="check" in words
    
    response=getTemplate(movingPlayer,pieceMoved,squareFrom,squareTo,capture=capture,capturedPiece=capturedPiece,check=check,castle=castle,castlingType=castlingType)
    
    return response



if __name__=="__main__":
    inputFile=open(sys.argv[1])
    outFile=open(sys.argv[1]+".template","w")
    for line in inputFile:
        response=getResponse(line)
        outFile.write(response+"\n")

    outFile.close()
