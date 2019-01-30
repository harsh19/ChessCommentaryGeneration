import chess
import chess.syzygy
#from boardInfo import boardInfo

# In[2]:

def getChessConstant(moveString):
    
    fileValue={"a":0, "b":1, "c":2, "d":3, "e":4 , "f":5, "g":6, "h":7}
    fileName=moveString[0]
    rankName=moveString[1]
    print moveString
    fileNum=fileValue[fileName]
    rankNum=int(rankName)-1
    
    chessConstant=8*rankNum+fileNum
    #print moveString
    print fileName
    print rankName
    print fileNum
    print rankNum
    print chessConstant
    return chessConstant

def convertToRowNotation(position):
    rankNum=position/8
    fileNum=position%8

    return 8*fileNum+rankNum

class BoardInfo:
    
    def __init__(self,initEngine=False):
        if initEngine:
            import chess.uci
            self.reinitEngine()

    def reinitEngine(self):
        self.engine = chess.uci.popen_engine("/home/vgangal/FinalProject/stockfish-8-linux/src/stockfish")
        self.engine.uci()
        self.info_handler=chess.uci.InfoHandler()
        self.engine.info_handlers.append(self.info_handler)

    def getScore(self,FENString):
        board=chess.Board(FENString)
        self.engine.position(board)
        self.engine.go(movetime=1000)
        score=self.info_handler.info["score"][1]
        return score

    def complement(self,string):
        if string=="black":
            return "white"
        else:
            return "black"
    def getAttackers(self,FENString,playerToMove,squareToMove,attacks=False):

        board=chess.Board(FENString)
        if playerToMove=="white":
            playerName=chess.WHITE
        else:
            playerName=chess.BLACK
    
        if not attacks:
            attackers=board.attackers(playerName,getChessConstant(squareToMove))
        else:
            attackers=board.attacks(getChessConstant(squareToMove))

        attackerIndices=[]
        for attacker in attackers:
            attackerIndices.append(convertToRowNotation(attacker))
        return attackerIndices
        #def getBestMoveAndScore():
        #def getBestFiveMovesAndScores():
        #def getBestMoveScore():

#board = chess.Board()


# In[3]:


#print "Board\n",board


# In[4]:


#board.push_san("e4")


# In[5]:


#print "Board\n",board


# In[6]:


#attackers = board.attackers(chess.WHITE, chess.F3)


# In[8]:


#print attackers



# In[10]:


#print "Attackers:",attackers
#print type(attackers)


# In[23]:




# In[14]:


#tablebases = chess.syzygy.open_tablebases("data/syzygy/regular")


# In[5]:



#board = chess.Board("1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1")


# In[10]:


#engine.position(board)
#engine.go(movetime=2000)


# In[11]:


#info_handler = chess.uci.InfoHandler()
#engine.info_handlers.append(info_handler)


# In[12]:


#engine.position(board)
#engine.go(movetime=1000)


# In[13]:


#info_handler.info["score"][1]


# In[14]:


#print "Board\n",board


# In[15]:


#board = chess.Board()
#engine.position(board)
#engine.go(movetime=1000)

#print info_handler.info["score"][1]


# In[ ]:


# sudo aptg-get install of stockfish
# install chess library

