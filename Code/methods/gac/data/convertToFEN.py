

def convertToFEN(pieceList,moveColor):

    pieceNameDict={"king":"k","queen":"q","knight":"n","rook":"r","bishop":"b","pawn":"p"}
    FENString=[]
    for chessAntiRank in range(8):
        rankString=[]
        chessRank=7-chessAntiRank
        for chessFile in range(8):
            #print chessFile
            #print chessRank
            index=8*chessFile+chessRank
            piece=pieceList[index]
            FENPiece=None
            if piece=="eps":
                FENPiece="1"
            else:
                words=piece.split("_")
                color=words[0]
                pieceName=words[1]
                FENPiece=pieceNameDict[pieceName]
                if color=="white":
                    FENPiece=FENPiece.upper()
            rankString.append(FENPiece)
        
        #print rankString
        index=0
        newRankString=[]
        while index<=7:
            if rankString[index]!="1":
                newRankString.append(rankString[index])
                index+=1
                #print "Continuing",index
            else:
                oneCount=0
                while index<=7 and rankString[index]=="1":
                    oneCount+=1
                    index+=1
                    #print "Expanding",index
                newRankString.append(str(oneCount))

        rankString="".join(newRankString)
        FENString.append(rankString)
        
    FENString="/".join(FENString)
    FENString=FENString+" "+moveColor+" - - 0 1"
    return FENString        



if __name__=="__main__":

    file1=["white_rook","eps","white_pawn","eps","eps","eps","black_pawn","black_rook"]
    file2=["eps","white_pawn","eps","eps","black_queen","black_bishop","black_pawn","eps"]
    file3=["eps",]*8
    file4=["eps",]*4+["white_pawn","black_pawn"]+["eps",]*2
    file5=["eps",]*7+["black_rook",]
    file6=["eps","white_pawn","white_queen","eps","white_knight","white_bishop","black_knight","eps"]
    file7=["eps","white_king"]+["eps",]*4+["white_pawn","black_king"]
    file8=["white_rook",]+["eps",]*5+["black_pawn","eps"]
    pieceList=file1+file2+file3+file4+file5+file6+file7+file8
    print len(pieceList)
    print convertToFEN(pieceList)
