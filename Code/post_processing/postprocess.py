
import en

opposite_player = {'black':'white', 'white':'black'}

def getOppositePlayer(player):
	return opposite_player[player]

def postProcess(s, player="black"):
	s = s.replace("...","")
	s = s.strip()
	s = s.replace("'m","am")
	words = s.split()
	print "words = ", words
	if words[0]=="so" or words[0]=="and":
		words = words[1:]
	if words[0].lower()=="i":
		if words[1].lower()=="think":
			#words[1] = "It"
			words = words[2:]
			#pass
		else:
			words[0] = str.upper(player[0]) + player[1:]
			opposite_player = getOppositePlayer(player)
			for j in range(len(words)):
				if words[j]=="his":
					words[j] = opposite_player #+ "'s"
			for j in range(len(words)):
				if words[j]=="my":
					words[j] = "his"
			if len(words)>1 and en.verb.tense(words[1])!="past":
				words[1] = en.verb.present(words[1], person=3, negate=False)
		s = " ".join(words)
	return s


if __name__=="__main__":
	s = "I protect my knight ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "I decide to exchange my bishop ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "I attack his queen with my knight ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "I attacked his queen with my knight ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "He attacked my queen with his knight ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "I 'm going to attack the bishop ."
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "... so I bring my knight out"
	print s, "\n", postProcess(s, player="black"), "\n"
	s = "i think it would have been better to play the bishop to move the knight ."
	print s, "\n", postProcess(s, player="black"), "\n"

	
