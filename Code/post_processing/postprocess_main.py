import postprocess
import sys

def getPieceColor(s):
	color = s.split("<EOP> ")[1].split()[0]
	return color

def getData(inp_fname, move_info_fname):
	inp = open(inp_fname,"r").readlines()
	inp = [row.strip() for row in inp]
	move_info = open(move_info_fname, "r").readlines()
	move_info = [getPieceColor(che) for che in move_info]
	assert len(move_info)==len(inp)
	return inp, move_info


def main(inp_fname, move_info_fname, output_fname):
	inp, move_info = getData(inp_fname, move_info_fname)
	fw = open(output_fname, "w")
	for txt,player in zip(inp, move_info):
		out = postprocess.postProcess(txt, player=player)
		fw.write(out + "\n")
	fw.close()


inp_fname = sys.argv[1]
move_info_fname = sys.argv[2]
output_fname = inp_fname + ".postprocessed"
main(inp_fname, move_info_fname, output_fname)