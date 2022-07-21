
import sys  
from PyQt4.QtGui import *  
from PyQt4.QtCore import *  
from PyQt4.QtWebKit import *  
from lxml import html 
import pickle
import time
from PyQt4 import QtGui, QtCore
import functools
import sys


import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-typ", dest="typ", help="home or subsequent", default='home')
    parser.add_argument("-i", type=int, dest="i", help="i")
    parser.add_argument("-num", type=int, dest="num", help="num")
    args = parser.parse_args()  
    return args
params = parseArguments()
#typ = params.typ


#Take this class for granted.Just use result of rendering.
class Render(QWebPage):  
  def __init__(self, url):  
	self.app = QApplication(sys.argv)  
	QWebPage.__init__(self)  
	self.loadFinished.connect(self._loadFinished)  
	qurl = QUrl(url)
	func = functools.partial(self.mainFrame().load, qurl )  
	timer = QtCore.QTimer()
	timer.timeout.connect(func)
	timer.start(10000)
	self.app.exec_()  
  
  def _loadFinished(self, result):  
	self.frame = self.mainFrame()  
	self.app.quit()  

def save_all():
	global cur_url
	global html_doc
	all_links = pickle.load( open("./saved_files/saved_links.p", "r") )
	#extra_links = pickle.load( open("extra_pages.p", "r") )
	print("len(all_links) = ",len(all_links))
	num = sys.argv[1]

	i = params.i
	print("i = ",type(i))
	num = params.num
	url = all_links[i]
	if num!=0:
		url+="&pg="+str(num)
	print("i, url = ",i,url)
	#This step is important.Converting QString to Ascii for lxml to process
	#archive_links = html.fromstring(str(result.toAscii()))
	
	cur_url = url
	error_count = 0
	try:
		r = Render(cur_url)
		result = r.frame.toHtml()
		html_doc = result.toAscii()
		
		if num==0:
			fw = open("./saved_files/saved"+str(i)+".html", "w")
		else:
			fw = open("./saved_files/saved"+str(i)+"_" + str(num) + ".html", "w")
		fw.write(html_doc)
		fw.close()
		print("---- SLEEPING ---- ")
		time.sleep(10)
	except:
		print("ERROR!!")
		error_count+=1
		print("error_count = ",error_count)
	##if i>4:
	##	break

if __name__=="__main__":
	save_all()

'''
s = "https://gameknot.com/annotation.pl/fierce-queen-taking-spanish-easy?gm=63368"

url = 'http://pycoders.com/archive/'  
url = s
r = Render(url)  
result = r.frame.toHtml()
#This step is important.Converting QString to Ascii for lxml to process
archive_links = html.fromstring(str(result.toAscii()))
print archive_links
print "---------"
print result.toAscii()
print "======================================"
print result

'''