from PerceptronBase import Perceptron
from util import read_file

class DepLabel(Perceptron):
	WORD, POS, HEAD, LABEL = 0, 1, 2, 3
	
	def __init__(self):
		self.total_labels = [
			'TMP', 'CJTN0', 'CJTN1', 'IO', 'SBJ', 'cCJTN', 'CJTN2', 'AUX', 
			'NMOD', 'COMP', 'OBJ', 'PRD', 'CND', 'CJTN4', 'LOC', 'UNK', 'OTHER',
			'CJT', 'CJTN', 'DIR', 'AMOD', 'BNF', 'ADV', 'FOC', 'MNR', 'APP', 'CJTN3',
			'PRP', 'DMOD', 'EXT', 'LGS', 'RELC', 'PRT', 'TPC'
		]	
		feature_names = [
			'p_word', 'p_pos', 's_word', 's_pos',
			'gp_word', 'gp_pos', 'gp_word_pos',
			'p_word_pos', 's_word_pos', 
			'p_s_words', 'p_s_poses',
			'gp_p_words', 'gp_p_poses',
			'gp_p_s_poses',
			'gp_children', 'p_children', 's_children',
			'p_s_distance', 'p_s_dir'
		]
		Perceptron.__init__(self, self.total_labels, feature_names)
		
	def get_children_num(self, sentence):
		children_num = [0] * len(sentence)
		for item in sentence:
			p = int(item[self.HEAD])
			if p != -1:
				children_num[p] += 1
		return children_num
		
	def extract_feature(self, sentence, children_num, s):
		s_len = len(sentence)
		p = int(sentence[s][self.HEAD])
		gp = int(sentence[p][self.HEAD])

		p_word = sentence[p][self.WORD]
		p_pos = sentence[p][self.POS]
		p_children = children_num[p]
		
		s_word = sentence[s][self.WORD]
		s_pos = sentence[s][self.POS]
		s_children = children_num[s]
			
		if gp != -1:
			gp_word = sentence[gp][self.WORD]
			gp_pos = sentence[gp][self.POS]
			gp_children = children_num[gp]
		else:
			gp_word = '_NULL_'
			gp_pos = '_NULL_'
			gp_children = -1
			
		feature = {}
		
		feature['p_word'] = p_word
		feature['p_pos'] = p_pos
		feature['p_word_pos'] = p_word + '|' + p_pos
		
		feature['s_word'] = s_word
		feature['s_pos'] = s_pos
		feature['s_word_pos'] = s_word + '|' + s_pos
		
		feature['gp_word'] = gp_word
		feature['gp_pos'] = gp_pos
		feature['gp_p_words'] = gp_word + '|' + p_word
		feature['gp_word_pos'] = gp_word + '|' + gp_pos
		
		feature['p_s_words'] = p_word + '|' + s_word
		feature['p_s_poses'] = p_pos + '|' + s_pos
		feature['gp_p_poses'] = gp_pos + '|' + p_pos
		
		feature['gp_children'] = gp_children
		feature['p_children'] = p_children
		feature['s_children'] = s_children
		
		feature['gp_p_s_poses'] = gp_pos + '|' + p_pos + '|' + s_pos
		
		if p > s:
			feature['p_s_dir'] = 'L'
			feature['p_s_distance'] = str(p-s)
		else:
			feature['p_s_dir'] = 'R'
			feature['p_s_distance'] = str(s-p)
		
		return feature
		
	def train(self, trn_file, iter_num = 10):
		all_feat_vec, all_labels = [], []
		
		sent_iter = read_file(trn_file)
		for sent in sent_iter:
			sent_feat_vec, labels = [], []
			children_num = self.get_children_num(sent)
			for idx in range(len(sent)):
				if sent[idx][self.HEAD] != '-1':
					feat_vec = self.extract_feature(sent, children_num, idx)
					sent_feat_vec.append(feat_vec)
					labels.append(sent[idx][self.LABEL])
			all_feat_vec.append(sent_feat_vec)
			all_labels.append(labels)
			
		print('--Training Perceptron for DepLabel--')
		self.train_t(all_feat_vec, all_labels, iter_num)
		
	def predict(self, sentence):
		predict_sentence = []
		children_num = self.get_children_num(sentence)
		
		for idx, item in enumerate(sentence):
			if item[self.HEAD] == '-1':
				predict_sentence.append([item[0], item[1], item[2], 'ROOT'])
			else:
				feat_vec = self.extract_feature(sentence, children_num, idx)
				predict_label = self.predict_t(feat_vec, self.total_labels)
				predict_sentence.append([item[0], item[1], item[2], predict_label])
				
		return predict_sentence
		
	def predict_and_save(self, infile, outfile):
		sent_iter = read_file(infile)
		handle = open(outfile, 'w', encoding='utf-8')
		for sent in sent_iter:
			predict_sentence = self.predict(sent)
			for item in predict_sentence:
				handle.write('{0}\t{1}\t{2}\t{3}\n'.format(item[0], item[1], item[2], item[3]))
			handle.write('\n')
		handle.close()
		
if __name__ == '__main__':
	depLabel = DepLabel()
	depLabel.train('d:\\Project1\\data\\trn.ec', 10)
	depLabel.predict_and_save('d:\\Project1\\data\\dev.ec', 'd:\\Project1\\ans3.ec')