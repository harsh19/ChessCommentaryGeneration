import json
import numpy as np
import os
import unicodedata
import pickle
import sys
import time

from utilities import Utilities

mode = sys.argv[1] # {expand, html_parser}

##############################################################

'''
- Input:
	- saved_links.p
- Go through the links
- Parse the html to see parginator
- Output
	- extra_pages.p : count of number of pages - in same order as in saved_links.p
'''
def expander_main():
	utils = Utilities()
	data_path = "./saved_files/"
	src = data_path + "saved_links.p"
	destination_path = data_path + "expanded_links.p"
	saved_links = pickle.load( open(src,"r") )
	counts = []
	for i, link in enumerate(saved_links):
		print "i,link = ",i, link
		num = 0
		#if i<2: continue
		#if i>3: break
		try:
			file = "saved"+str(i)+".html"
			html_doc = open(data_path + file,"r").read()
			soup = utils.getSoupFromHTML(html_doc)
			paginator = utils.getTableOfClass(soup, 'paginator')
			print "len(paginator) = ",len(paginator)
			if len(paginator)>0:
				paginator = paginator[0] # If it occurs, it occurs twice - and both seem to be identical
				txt = utils.soupToText(paginator).lower()
				if txt.count("next")>0:
					txt = txt.replace("pages:","").strip()
					for j,c in enumerate(txt):
						if c>='1' and c<='9': # assuming <=9 extra pages
							num+=1
						else:
							break
			print "num = ",num
		except:
			print "----ERROR:"
		counts.append(num)
	print "len(counts) = ",len(counts)
	print "sum(counts) = ",sum(counts)
	pickle.dump(counts, open("extra_pages.p","w"))
			


			


##############################################################
class DataCollector:

	def __init__(self):
		self._utils = Utilities()
		self._data_path = "./saved_files/"
		self._destination_path = "./outputs/"
		print "------"

	def _getList(self):
		#files = open(self._data_path)
		files = os.listdir(self._data_path)
		files = [f for f in files if f.count("html")>0]
		#return ["saved0.html"] # debug mode
		return files

	def _getBoardValues(self, soup):
		divs = self._utils.getDivOfClass(soup, "cdiag_frame") # only 1 div - use first one
		#print "len(divs) = ",len(divs) ## should be 1
		for div in divs:
			board = self._utils.getDivOfID(div, "board")
			board_elements = self._utils.getDivAll(board[0], recursive=False)
			#board_labels = self._utils.getTableOfClass(soup, 'boardlabel') ##-- Not Needed - the backen data structure seems to be consisten in black and white views
			#print "board_labels = ",board_labels
			#print len(board_elements)
			board_element_vals = []
			for ele in board_elements:
				tmp = {}
				ele_style_vals = ele['style'].split(';')
				ele_style_vals = {val.split(": ")[0].strip():val.split(": ")[1].strip().replace("px","") for val in ele_style_vals if val.count(": ")>0}
				if 'left' in ele_style_vals:
					left =  ele_style_vals['left']
					top = ele_style_vals['top']
					tmp['left'] = left
					tmp['top'] = top
					ele_img = self._utils.getImgAll(ele)
					if len(ele_img)>0:
						ele_img = ele_img[0]
						ele_img_style = ele_img['style']
						ele_style_vals = ele_img_style.split(';')
						ele_style_vals = {val.split(": ")[0].strip():val.split(": ")[1].strip().replace("px","") for val in ele_style_vals if val.count(": ")>0}
						if 'left' in ele_style_vals:
							left_img =  ele_style_vals['left']
							top_img = ele_style_vals['top']
							tmp['left_img'] = left_img
							tmp['top_img'] = top_img
					board_element_vals.append(tmp)
					#print tmp
			#print len(board_element_vals)
			return board_element_vals

	def _boardCellToInfo(self, list_of_board_cells):
		ret = []
		column_names = ['a','b','c','d','e','f','g','h']
		column_name_index = 0
		row_names = ['1','2','3','4','5','6','7','8']
		row_name_index = 0
		for cell in list_of_board_cells:
			cur = {}
			top = cell['top']
			left = cell['left']
			cell_location = column_names[column_name_index] + row_names[row_name_index]
			row_name_index+=1
			if row_name_index == len(row_names):
				row_name_index=0
				column_name_index+=1
			cur['location'] = cell_location
			if 'left_img' in cell:
				piece = ""
				left_image = cell['left_img']
				top_image = cell['top_img']
				left = int(left_image)
				if left==0:
					piece="white"
				else:
					piece = "black"
				top_image_mapping = {
				'0': "king",
				'-30': "queen" , 
				'-60' : "rook" ,
				'-90' : "knight" ,
				'-120' : "bishop" ,
				'-150' : "pawn" }
				piece = piece + "_" + top_image_mapping[top_image]
				cur['piece'] = piece
			ret.append(cur)
		return ret

	def getData(self):
		all_files = self._getList()
		fw = open("error_files.txt","w")
		for file in all_files:
			try:
				print "file = ",file
				html_doc = open(self._data_path + file,"r").read()
				soup = self._utils.getSoupFromHTML(html_doc)
				results = soup.findAll("table",{"class":"dialog"})
				tmp = results[0] # expecting only 1 table of this type
				results2 = tmp.findAll("tr")
				results2 = [ result for result in results2 if len(result.findAll("td", recursive=False))==2 and len(self._utils.getDivOfClass(result, "cdiag_frame"))>0 ] #based on observation
				all_steps_info = []
				for index,result in enumerate(results2):
					if index%2==1:
						continue #fix for repetitions
					#print "result = ",result
					td_res = result.findAll("td", recursive=False)
					td = td_res[0] ## move+board
					##--- Extract moves
					txt =  td.get_text()
					txt = unicodedata.normalize('NFKD', txt).encode('ascii','ignore')
					moves = txt[:txt.find("<!--")].strip()
					##--- Extract board elements
					board_element_vals = self._getBoardValues(result)
					board_element_info = self._boardCellToInfo(board_element_vals)
					#for info in board_element_info:
					#	print info
					##--- Get the comment
					td = td_res[1] ## Comment
					comment = self._utils.soupToText(td)
					##--- Add to data structure
					current_step_info = [moves, board_element_info, comment]
					all_steps_info.append(current_step_info)
				pickle.dump( all_steps_info, open(self._destination_path + file.replace(".html",".obj"), "w") )
			except:
				fw.write(file)
				fw.write("\n")
		fw.close()
		#print "=========================================================="

##############################################################

if mode=="html_parser":
	data_collector = DataCollector()
	data_collector.getData()
elif mode=="expand":
	expander_main()
else:
	print "Wrong option"
