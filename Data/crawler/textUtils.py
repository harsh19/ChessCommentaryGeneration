import math

class textUtils():

    def isMove(self,word):
        #Replace this by a regex later
        if "..." in word or (word[-1].isdigit() and len(word)>=3):
            return True
        else:
            return False

