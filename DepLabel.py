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
			'p_word_pos', 's_word_pos', 
			'p_s_words', 'p_s_poses',
			'qp_word', 'qp_pos', 'np_word', 'np_pos',
			'qs_word', 'qs_pos', 'ns_word', 'ns_pos',
			'qp_word_pos', 'np_word_pos', 'qs_word_pos', 'ns_word_pos',
			'qp_p_poses', 'p_np_poses',
			'qs_s_poses', 's_ns_poses',
			'qp_p_np_poses', 'qs_s_ns_poses',
			'p_s_distance', 'p_s_dir'
		]
		Perceptron.__init__(self, self.total_labels, feature_names)
		
	def extract_feature(self, sentence, s):
		s_len = len(sentence)
		p = int(sentence[s][self.HEAD])
		
		if p > 0:
			qp_word = sentence[p-1][self.WORD]
			qp_pos = sentence[p-1][self.POS]
		else:
			qp_word = '_NULL_'
			qp_pos = '_NULL_'	
		p_word = sentence[p][self.WORD]
		p_pos = sentence[p][self.POS]	
		if p+1 < s_len:
			np_word = sentence[p+1][self.WORD]
			np_pos = sentence[p+1][self.POS]
		else:
			np_word = '_NULL_'
			np_pos = '_NULL_'
			
		if s > 0:
			qs_word = sentence[s-1][self.WORD]
			qs_pos = sentence[s-1][self.POS]
		else:
			qs_word = '_NULL_'
			qs_pos = '_NULL_'			
		s_word = sentence[s][self.WORD]
		s_pos = sentence[s][self.POS]			
		if s+1 < s_len:
			ns_word = sentence[s+1][self.WORD]
			ns_pos = sentence[s+1][self.POS]
		else:
			ns_word = '_NULL_'
			ns_pos = '_NULL_'
			
		feature = {}
		
		feature['qp_word'] = qp_word
		feature['p_word'] = p_word
		feature['np_word'] = np_word
		feature['qp_pos'] = qp_pos
		feature['p_pos'] = p_pos
		feature['np_pos'] = np_pos
		
		feature['qs_word'] = qs_word
		feature['s_word'] = s_word
		feature['ns_word'] = ns_word
		feature['qs_pos'] = qs_pos
		feature['s_pos'] = s_pos
		feature['ns_pos'] = ns_pos
		
		feature['p_s_words'] = p_word + '|' + s_word
		feature['p_s_poses'] = p_pos + '|' + s_pos
		
		feature['qp_word_pos'] = qp_word + '|' + qp_pos
		feature['p_word_pos'] = p_word + '|' + p_pos
		feature['np_word_pos'] = np_word + '|' + np_pos
		
		feature['qs_word_pos'] = qs_word + '|' + qs_pos
		feature['s_word_pos'] = s_word + '|' + s_pos
		feature['ns_word_pos'] = ns_word + '|' + ns_pos
		
		feature['qp_p_poses'] = qp_pos + '|' + p_pos
		feature['p_np_poses'] = p_pos + '|' + np_pos
		feature['qs_s_poses'] = qs_pos + '|' + s_pos
		feature['s_ns_poses'] = s_pos + '|' + ns_pos
		feature['qp_p_np_poses'] = qp_pos + '|' + p_pos + '|' + np_pos
		feature['qs_s_ns_poses'] = qs_pos + '|' + s_pos + '|' + ns_pos
		
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
			for idx in range(len(sent)):
				if sent[idx][self.HEAD] != '-1':
					feat_vec = self.extract_feature(sent, idx)
					sent_feat_vec.append(feat_vec)
					labels.append(sent[idx][self.LABEL])
			all_feat_vec.append(sent_feat_vec)
			all_labels.append(labels)
			
		print('--Training Perceptron for DepLabel--')
		self.train_t(all_feat_vec, all_labels, iter_num)
		
	def predict(self, sentence):
		predict_sentence = []
		
		for idx, item in enumerate(sentence):
			if item[self.HEAD] == '-1':
				predict_sentence.append([item[0], item[1], item[2], 'ROOT'])
			else:
				feat_vec = self.extract_feature(sentence, idx)
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
	depLabel.train('d:\\Project1\\data\\trn.ec')
	depLabel.predict_and_save('d:\\Project1\\data\\dev.ec', 'd:\\Project1\\result_3.ec')