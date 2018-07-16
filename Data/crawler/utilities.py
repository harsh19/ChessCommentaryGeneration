import urllib2
from bs4 import BeautifulSoup
import json

class Utilities:
	def getSoupFromURL(self,url):
		hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11', 'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8', 'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3', 'Accept-Encoding': 'none', 'Accept-Language': 'en-US,en;q=0.8', 'Connection': 'keep-alive'}
		req = urllib2.Request(url, headers=hdr)
		page = urllib2.urlopen(req)
		html_doc = page.read()
		soup = BeautifulSoup(html_doc, "lxml")
		return soup
		
	def getSoupFromHTML(self,html_doc):
		soup = BeautifulSoup(html_doc, "lxml")
		return soup

	def soupToText(self,ele):
	   tmp = [s.extract() for s in ele(['style', 'script', '[document]', 'head', 'title'])]
	   return ele.get_text()

	def getDivOfClass(self, soup, class_value, recursive=True):
		mydivs = soup.findAll("div", { "class" : class_value }, recursive=recursive)
		return mydivs

	def getDivAll(self, soup, recursive=True):
		mydivs = soup.findAll("div", recursive=recursive)
		return mydivs

	def getTableOfClass(self, soup, class_value, recursive=True):
		mydivs = soup.findAll("table", { "class" : class_value }, recursive=recursive)
		return mydivs

	def getImgAll(self, soup):
		mydivs = soup.findAll("img")
		return mydivs

	def getDivOfID(self, soup, id_value, recursive=True):
		mydivs = soup.findAll("div", { "id" : id_value }, recursive=recursive)
		return mydivs

	def getHREF(self,soup,class_value):
		mydivs = soup.findAll("div", { "class" : class_value })
		ret = []
		for mydiv in mydivs:
			mydiv_a = mydiv.find("a", { "class" : "rg_l" })
			ret.append(mydiv_a['href'])
		return ret

	def getAsciiOnly(self,txt):
		ret = ''.join([ch for ch in txt if ord(ch)<=128])
		return ret
		


'''
	def getTextClass(self,url):
		soup = self.getSoup(url)
		mydivs = soup.findAll("div", { "class" : "section-inner layoutSingleColumn" })
		ret = ""
		for mydiv in mydivs:
			ret = ret + soupToText(mydiv)
		return ret
'''