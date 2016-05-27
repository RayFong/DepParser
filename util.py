def read_file(filename):
	handle = open(filename, 'r', encoding='utf-8')
	sentence = []
	for l in handle:
		line = l.strip()
		if line:
			items = line.split('\t')
			sentence.append(items)
		else:
			yield sentence
			sentence = []
	handle.close()
	
def check_EC_cat(filename):
	print("Check the EC categary")
	sent_iter = read_file(filename)
	cate = set()
	for sent in sent_iter:
		cnt = 0
		for item in sent:
			if item[0] == '*PRO*':
				cnt += 1
			else:
				cate.add(cnt)
				cnt = 0
		if cnt > 0:
			print('The PRO is in the last')
				
	print(cate)
	
def check_dep_label(filename):
	print("Chcek the dependency label")
	sent_iter = read_file(filename)
	cate = set()
	for sent in sent_iter:
		for item in sent:
			cate.add(item[-1])
	print(cate)
	
def convert_to_no_EC(raw_sentence):
	sentence = []
	for item in raw_sentence:
		if item[0] != '*PRO*':
			sentence.append(item)
	return sentence
	
if __name__ == '__main__':
	check_dep_label('d:\\Project1\\data\\trn.ec')
	check_EC_cat('d:\\Project1\\data\\trn.ec')