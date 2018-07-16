


##### create combined files of desc, qual, 2comparative  
def _createCombineWithCategories(inp_file_types, feat_type, split, output_type):
	data_path = "data/"
	"test.che-eng.0attack.en"
	fw_en = open(data_path + split + ".che-eng." + output_type + feat_type + ".en", "w" )
	fw_che = open(data_path + split + ".che-eng." + output_type + feat_type + ".che", "w" )
	fw_cat = open(data_path + split + ".che-eng." + output_type + feat_type + ".categories", "w" )
	
	for inp_type in inp_file_types:
	
		data = open(data_path + split + ".che-eng." + inp_type + feat_type + ".en", "r").readlines()
		for row in data:
			fw_en.write(row.strip() + "\n")
	
		data = open(data_path + split + ".che-eng." + inp_type + feat_type + ".che", "r").readlines()
		for row in data:
			fw_che.write(row.strip() + "\n")
	
		m = len(data)
		for row in range(m):
			fw_cat.write(inp_type + "\n")

	fw_en.close()
	fw_che.close()
	fw_cat.close()

def createCombineWithCategories():
	feat_types = ["simple","attack","score"]
	splits = ["train", "valid", "test"]
	output_type = "7"
	inp_file_types = ["0","1","2.comparitive"]
	for feat_type in feat_types:
		for split in splits:
			_createCombineWithCategories(inp_file_types, feat_type, split, output_type)
			#break
		#break

######







createCombineWithCategories()