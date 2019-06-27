import pickle
import sys
import subprocess
import os
import time
from subprocess import call

processes = set()
command = 'python classify.py'
max_processes = int(sys.argv[3])

start = int(sys.argv[1])
end =   int(sys.argv[2])

all_links = pickle.load( open("./saved_files/saved_links.p", "r") )
extra_links = pickle.load( open("extra_pages.p", "r") )
for i, link in enumerate(all_links):
	if i<start: continue
	if i>end: break
	num = extra_links[i]
	print "i,num: ",i,num # 0 is no extra, if extra then (num-1) extra pages
	for j in range(0,num): # j=0 is the home page. Begin range from 0 if home page is to be included.
		print j
		processes.add( subprocess.Popen(["python", "save_rendered_webpage.py", "-i", str(i), "-num", str(j) ]) )
		if len(processes) >= max_processes:
			print "=========== Waiting"
			os.wait()
			print "========== Waiting ends"
			processes.difference_update([p for p in processes if p.poll() is not None])
