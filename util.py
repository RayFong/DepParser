from collections import defaultdict

# 读取文件
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

# 检查每个token前面空节点个数的种类	
def check_EC_cat(filename):
	print("Check the EC categary")
	sent_iter = read_file(filename)
	cate = defaultdict(int)
	for sent in sent_iter:
		cnt = 0
		for item in sent:
			if item[0] == '*PRO*':
				cnt += 1
			else:
				cate[cnt] += 1
				cnt = 0
		if cnt > 0:
			print('The PRO is in the last')
				
	for k, v in cate.items():
		print('{0}\t{1}'.format(k, v))

# 检查所有的依存关系描述符	
def check_dep_label(filename):
	print("Chcek the dependency label")
	sent_iter = read_file(filename)
	cate = defaultdict(int)
	for sent in sent_iter:
		for item in sent:
			cate[item[-1]] += 1
	for k, v in cate.items():
		print('{0}\t{1}'.format(k, v))

# 将带空节点的句子转成不带空节点的句子	
def convert_to_no_EC(raw_sentence):
	sentence = []
	for item in raw_sentence:
		if item[0] != '*PRO*':
			sentence.append(item)
	return sentence

# 提取每个序列的空节点	
def get_sent_ec(sent):
	ecs, cnt = [], 0
	for item in sent:
		if item[0] == '*PRO*':
			cnt += 1
		else:
			ecs.append(cnt)
			cnt = 0
	return ecs

# 计算预测ECs的准确率以及回召率	
def cal_ecs_acc_rec(file1, file2):
	sent_iter_1 = read_file(file1)
	sent_iter_2 = read_file(file2)
	word_acc, word_rec = [0, 0], [0, 0]
	sent_acc = [0, 0]
	
	for (sent1, sent2) in zip(sent_iter_1, sent_iter_2):
		ecs1 = get_sent_ec(sent1)
		ecs2 = get_sent_ec(sent2)
		valid = 1
		for (e1, e2) in zip(ecs1, ecs2):
			if e1 > 0:
				word_acc[0] += (e1 == e2)
				word_acc[1] += 1
			# word_acc[0] += (e1 == e2)
			# word_acc[1] += 1
			if e2 > 0:
				word_rec[0] += (e1 == e2)
				word_rec[1] += 1
			if e1 != e2:
				valid = 0
		sent_acc[0] += valid
		sent_acc[1] += 1
	
	word_a = word_acc[0] / float(word_acc[1])
	word_r = word_rec[0] / float(word_rec[1])
	sent_a = sent_acc[0] / float(sent_acc[1])
	print('word accuracy: %f (%d/%d)' % (word_a, word_acc[0], word_acc[1]))
	print('word recall: %f (%d/%d)' % (word_r, word_rec[0], word_rec[1]))
	print('sentence accuracy: %f (%d/%d)' % (sent_a, sent_acc[0], sent_acc[1]))

# 计算transition_based parsing的准确度
def compute_transition_accuracy(data, model):
	data_iter = read_file(data)
	model_iter = read_file(model)
	word_acc, sent_acc = [0, 0], [0, 0]
	for (sent1, sent2) in zip(data_iter, model_iter):
		valid = 1
		for (item1, item2) in zip(sent1, sent2):
			if item1[2] == item2[2]:
				word_acc[0] += 1
			else:
				valid = 0
			word_acc[1] += 1
		sent_acc[0] += valid
		sent_acc[1] += 1
	word_a = word_acc[0] / float(word_acc[1])
	sent_a = sent_acc[0] / float(sent_acc[1])
	print('word accuracy: %f (%d/%d)' % (word_a, word_acc[0], word_acc[1]))
	print('sentence accuracy: %f (%d/%d)' % (sent_a, sent_acc[0], sent_acc[1]))
	
# 计算预测依存关系描述符的准确度
def compute_deplabel_accuracy(data, model):
	data_iter = read_file(data)
	model_iter = read_file(model)
	word_acc, sent_acc = [0, 0], [0, 0]
	for (sent1, sent2) in zip(data_iter, model_iter):
		valid = 1
		for (item1, item2) in zip(sent1, sent2):
			if item1[3] == item2[3]:
				word_acc[0] += 1
			else:
				valid = 0
			word_acc[1] += 1
		sent_acc[0] += valid
		sent_acc[1] += 1
	word_a = word_acc[0] / float(word_acc[1])
	sent_a = sent_acc[0] / float(sent_acc[1])
	print('word accuracy: %f (%d/%d)' % (word_a, word_acc[0], word_acc[1]))
	print('sentence accuracy: %f (%d/%d)' % (sent_a, sent_acc[0], sent_acc[1]))
	
if __name__ == '__main__':
	check_dep_label('d:\\Project1\\data\\trn.ec')
	# check_EC_cat('d:\\Project1\\data\\trn.ec')
	# cal_ecs_acc_rec('d:\\Project1\\ans.ec', 'd:\\Project1\\data\\dev.ec')
	# compute_transition_accuracy('d:\\Project1\\ans2.ec', 'd:\\Project1\\data\\dev.ec')
	# compute_deplabel_accuracy('d:\\Project1\\ans3.ec', 'd:\\Project1\\data\\dev.ec')