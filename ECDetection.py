from PerceptronBase import Perceptron
from util import read_file, convert_to_no_EC

class ECDetection(Perceptron):
	WORD, POS = 0, 1
	
	def __init__(self):
		categray = [0, 1, 2, 3, 4]
		feature_names = [
			'pp_word', 'pp_pos', 'p_word', 'p_pos',
			'c_word', 'c_pos',
			'n_word', 'n_pos', 'nn_word', 'nn_pos',
			'pp_word_pos', 'p_word_pos', 
			'c_word_pos',
			'n_word_pos', 'nn_word_pos',
			'pp_p_words', 'pp_p_poses',
			'p_c_words', 'p_c_poses',
			'c_n_words', 'c_n_poses',
			'n_nn_words', 'n_nn_poses',
			'pp_p_c_poses', 'p_c_n_poses', 'c_n_nn_poses'
		]
		
		Perceptron.__init__(self, categray, feature_names)
		
	def get_labels(self, raw_sentence):
		sentence, labels = [], []
		cnt = 0
		for item in raw_sentence:
			if item[self.WORD] == '*PRO*':
				cnt += 1
			else:
				sentence.append([item[self.WORD], item[self.POS]])
				labels.append(cnt)
				cnt = 0
				
		return sentence, labels
		
	def extract_feature(self, sentence, i):
		s_len = len(sentence)
		
		if i > 1:
			pp_word = sentence[i-2][self.WORD]
			pp_pos = sentence[i-2][self.POS]
		else:
			pp_word = '_NULL_'
			pp_pos = '_NULL_'
		if i > 0:
			p_word = sentence[i-1][self.WORD]
			p_pos = sentence[i-1][self.POS]
		else:
			p_word = '_NULL_'
			p_pos = '_NULL_'
			
		c_word = sentence[i][self.WORD]
		c_pos = sentence[i][self.POS]
		
		if i+1 < s_len:
			n_word = sentence[i+1][self.WORD]
			n_pos = sentence[i+1][self.POS]
		else:
			n_word = '_NULL_'
			n_pos = '_NULL_'
		if i+2 < s_len:
			nn_word = sentence[i+2][self.WORD]
			nn_pos = sentence[i+2][self.POS]
		else:
			nn_word = '_NULL_'
			nn_pos = '_NULL_'
			
		feature = {}
		feature['pp_word'] = pp_word
		feature['pp_pos'] = pp_pos
		feature['p_word'] = p_word
		feature['p_pos'] = p_pos
		feature['c_word'] = c_word
		feature['c_pos'] = c_pos
		feature['n_word'] = n_word
		feature['n_pos'] = n_pos
		feature['nn_word'] = nn_word
		feature['nn_pos'] = nn_pos
		
		feature['pp_word_pos'] = pp_word + '|' + pp_pos
		feature['p_word_pos'] = p_word + '|' + p_pos
		feature['c_word_pos'] = c_word + '|' + c_pos
		feature['n_word_pos'] = n_word + '|' + n_pos
		feature['nn_word_pos'] = nn_word + '|' + nn_pos
		
		feature['pp_p_words'] = pp_word + '|' + p_word
		feature['p_c_words'] = p_word + '|' + c_word
		feature['c_n_words'] = c_word + '|' + n_word
		feature['n_nn_words'] = n_word + '|' + nn_word
		
		feature['pp_p_poses'] = pp_pos + '|' + p_pos
		feature['p_c_poses'] = p_pos + '|' + c_pos
		feature['c_n_poses'] = c_pos + '|' + n_pos
		feature['n_nn_poses'] = n_pos + '|' + nn_pos
		
		feature['pp_p_c_poses'] = pp_pos + '|' + p_pos + '|' + c_pos
		feature['p_c_n_poses'] = p_pos + '|' + c_pos + '|' + n_pos
		feature['c_n_nn_poses'] = c_pos + '|' + n_pos + '|' + nn_pos
		
		return feature
		
	def train(self, trn_file, iter_num = 10):
		all_feat_vec, all_labels = [], []
		
		sent_iter = read_file(trn_file)
		for sent in sent_iter:
			sentence, labels = self.get_labels(sent)
			sent_feat_vec = []
			for idx in range(len(sentence)):
				feat_vec = self.extract_feature(sentence, idx)
				sent_feat_vec.append(feat_vec)
				
			all_feat_vec.append(sent_feat_vec)
			all_labels.append(labels)
			
		print('--Training Perceptron for EC--')
		self.train_t(all_feat_vec, all_labels, iter_num)
		
	def predict(self, sentence):
		predict_sentence = []
		categray = [0, 1, 2, 3, 4]
		s_len = len(sentence)
		
		for idx in range(s_len):
			feat_vec = self.extract_feature(sentence, idx)
			predict_cate = self.predict_t(feat_vec, categray)
			while predict_cate > 0:
				predict_sentence.append(['*PRO*', 'EMCAT'])
				predict_cate -= 1
			predict_sentence.append(sentence[idx])
			
		return predict_sentence
		
	def predict_and_save(self, infile, outfile):
		sent_iter = read_file(infile)
		handle = open(outfile, 'w', encoding='utf-8')
		for sent in sent_iter:
			predict_sentence = self.predict(sent)
			for item in predict_sentence:
				handle.write('{0}\t{1}\n'.format(item[0], item[1]))
			handle.write('\n')
		handle.close()
		
	def test_dev(self, infile, outfile):
		sent_iter = read_file(infile)
		handle = open(outfile, 'w', encoding='utf-8')
		for sent in sent_iter:
			predict_sentence = self.predict(convert_to_no_EC(sent))
			for item in predict_sentence:
				handle.write('{0}\t{1}\n'.format(item[0], item[1]))
			handle.write('\n')
		handle.close()
		
if __name__ == '__main__':
	detection = ECDetection()
	detection.train('d:\\Project1\\data\\trn.ec', 15)
	detection.test_dev('d:\\Project1\\data\\dev.ec', 'd:\\Project1\\result_2.ec')
	