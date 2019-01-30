import sys
src = sys.argv[1]
data = open(src,"r").readlines()
fw = open(src+".lower","w")
for row in data:
    row =  row.strip().lower()
    fw.write(row+"\n")
fw.close()