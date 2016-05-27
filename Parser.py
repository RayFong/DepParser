from PerceptronBase import Perceptron
from util import read_file

class Parser(Perceptron):
	INDEX, WORD, POS, LEFTNODE, RIGHTNODE = 0, 1, 2, 3, 4
	
	def __init__(self):
		labels = ['LA', 'RA', 'SH']
		feature_names = [
			'pp_word', 'pp_pos', 'p_word', 'p_pos',
			's_word', 's_pos',
			's_l_word', 's_l_pos', 's_r_word', 's_r_pos',
			'q_word', 'q_pos',
			'q_l_word', 'q_l_pos', 'q_r_word', 'q_r_pos',
			'nn_word', 'nn_pos', 'n_word', 'n_pos',
			'pp_word_pos', 'p_word_pos',
			's_word_pos', 'q_word_pos',
			'n_word_pos', 'nn_word_pos',
			'pp_p_words', 'pp_p_poses',
			'p_s_words', 'p_s_poses',
			's_q_words', 's_q_poses', 
			'q_n_words', 'q_n_poses',
			'n_nn_words', 'n_nn_poses',  
			'pp_p_s_poses', 'p_s_q_poses', 's_q_n_poses', 'q_n_nn_poses',
			's_q_distance'
		]
		
		Perceptron.__init__(self, labels, feature_names)
		
	def get_tree(self, sentence):
		vertexs, arcs = [], []
		for idx, item in enumerate(sentence):
			vertexs.append([idx, item[0], item[1], None, None])
			arcs.append((int(item[2]), idx))
		return vertexs, arcs
		
	def extract_feature(self, stack, queue):
		s_len = len(stack)
		q_len = len(queue)
		
		# 靠近栈顶的两个元素的word及pos
		if s_len > 2:
			pp_word = stack[-3][self.WORD]
			pp_pos = stack[-3][self.POS]
		else:
			pp_word = '_NULL_'
			pp_pos = '_NULL_'
		if s_len > 1:
			p_word = stack[-2][self.WORD]
			p_pos = stack[-2][self.POS]
		else:
			p_word = '_NULL_'
			p_pos = '_NULL_'
		
		# 栈顶元素的相关特征
		if s_len > 0:
			s_word = stack[-1][self.WORD]
			s_pos = stack[-1][self.POS]
			if stack[-1][self.LEFTNODE] != None:
				s_l_word = stack[-1][self.LEFTNODE][self.WORD]
				s_l_pos = stack[-1][self.LEFTNODE][self.POS]
			else:
				s_l_word = '_NULL_'
				s_l_pos = '_NULL_'
			if stack[-1][self.RIGHTNODE] != None:
				s_r_word = stack[-1][self.RIGHTNODE][self.WORD]
				s_r_pos = stack[-1][self.RIGHTNODE][self.POS]
			else:
				s_r_word = '_NULL_'
				s_r_pos = '_NULL_'
		else:
			s_word = '_NULL_'
			s_pos = '_NULL_'
			s_l_word = '_NULL_'
			s_l_pos = '_NULL_'
			s_r_word = '_NULL_'
			s_r_pos = '_NULL_'
			
		# 队列队首元素的相关特征
		if q_len > 0:
			q_word = queue[0][self.WORD]
			q_pos = queue[0][self.POS]
			if queue[0][self.LEFTNODE] != None:
				q_l_word = queue[0][self.LEFTNODE][self.WORD]
				q_l_pos = queue[0][self.LEFTNODE][self.POS]
			else:
				q_l_word = '_NULL_'
				q_l_pos = '_NULL_'
			if queue[0][self.RIGHTNODE] != None:
				q_r_word = queue[0][self.RIGHTNODE][self.WORD]
				q_r_pos = queue[0][self.RIGHTNODE][self.POS]
			else:
				q_r_word = '_NULL_'
				q_r_pos = '_NULL_'
		else:
			q_word = '_NULL_'
			q_pos = '_NULL_'
			q_l_word = '_NULL_'
			q_l_pos = '_NULL_'
			q_r_word = '_NULL_'
			q_r_pos = '_NULL_'
			
		# 靠近队首两个元素的特征
		if q_len > 1:
			n_word = queue[1][self.WORD]
			n_pos = queue[1][self.POS]
		else:
			n_word = '_NULL_'
			n_pos = '_NULL_'
		if q_len > 2:
			nn_word = queue[2][self.WORD]
			nn_pos = queue[2][self.POS]
		else:
			nn_word = '_NULL_'
			nn_pos = '_NULL_'
			
		# 距离特征
		if s_len > 0 and q_len > 0:
			s_q_distance = str(queue[0][self.INDEX] - stack[-1][self.INDEX])
		else:
			s_q_distance = '_NULL_'
			
		# 赋值特征
		feature = {}
		feature['pp_word'] = pp_word
		feature['pp_pos'] = pp_pos
		feature['p_word'] = p_word
		feature['p_pos'] = p_pos
		feature['s_word'] = s_word
		feature['s_pos'] = s_pos
		feature['s_l_word'] = s_l_word
		feature['s_l_pos'] = s_l_pos
		feature['s_r_word'] = s_r_word
		feature['s_r_pos'] = s_r_pos
		feature['q_word'] = q_word
		feature['q_pos'] = q_pos
		feature['q_l_word'] = q_l_word
		feature['q_l_pos'] = q_l_pos
		feature['q_r_word'] = q_r_word
		feature['q_r_pos'] = q_r_pos
		feature['n_word'] = n_word
		feature['n_pos'] = n_pos
		feature['nn_word'] = nn_word
		feature['nn_pos'] = nn_pos
		
		feature['pp_word_pos'] = pp_word + '|' + pp_pos
		feature['p_word_pos'] = p_word + '|' + p_pos
		feature['s_word_pos'] = s_word + '|' + s_pos
		feature['q_word_pos'] = q_word + '|' + q_pos
		feature['n_word_pos'] = n_word + '|' + n_pos
		feature['nn_word_pos'] = nn_word + '|' + nn_pos
		
		feature['pp_p_words'] = pp_word + '|' + p_word
		feature['p_s_words'] = p_word + '|' + s_word
		feature['s_q_words'] = s_word + '|' + q_word
		feature['q_n_words'] = q_word + '|' + n_word
		feature['n_nn_words'] = n_word + '|' + nn_word
		feature['pp_p_poses'] = pp_pos + '|' + p_pos
		feature['p_s_poses'] = p_pos + '|' + s_pos
		feature['s_q_poses'] = s_pos + '|' + q_pos
		feature['q_n_poses'] = q_pos + '|' + n_pos
		feature['n_nn_poses'] = p_pos + '|' + pp_pos
		
		feature['pp_p_s_poses'] = pp_pos + '|' + p_pos + '|' + s_pos
		feature['p_s_q_poses'] = p_pos + '|' + s_pos + '|' + q_pos
		feature['s_q_n_poses'] = s_pos + '|' + q_pos + '|' + n_pos
		feature['q_n_nn_poses'] = q_pos + '|' + n_pos + '|' + nn_pos
		
		feature['s_q_distance'] = s_q_distance
		
		return feature
		
	def get_sent_feat_vec(self, sentence):
		stack, solved_arcs = [[-1, '_ROOT_', '_ROOT_', None, None]], []
		queue, arcs = self.get_tree(sentence)
		sent_feat_vec, labels = [], []
		
		while queue != []:
			has_arc = False
			if stack != [] and (queue[0][self.INDEX], stack[-1][self.INDEX]) in arcs:
				labels.append('LA')
				sent_feat_vec.append(self.extract_feature(stack, queue))
				solved_arcs.append((queue[0][self.INDEX], stack[-1][self.INDEX]))
				queue[0][self.LEFTNODE] = stack[-1]
				stack.pop()
				has_arc = True
			elif stack != [] and (stack[-1][self.INDEX], queue[0][self.INDEX]) in arcs:
				valid = True
				for arc in arcs:
					if queue[0][self.INDEX] == arc[0] and arc not in solved_arcs:
						valid = False
						break
				if valid:
					labels.append('RA')
					sent_feat_vec.append(self.extract_feature(stack, queue))
					solved_arcs.append((stack[-1][self.INDEX], queue[0][self.INDEX]))
					stack[-1][self.RIGHTNODE] = queue[0]
					queue[0] = stack.pop()
					has_arc = True
					
			if not has_arc:
				labels.append('SH')
				sent_feat_vec.append(self.extract_feature(stack, queue))
				stack.append(queue[0])
				del queue[0]
				
		return sent_feat_vec, labels
		
	def train(self, trn_file, iter_num = 15):
		all_feat_vec, all_labels = [], []
		
		sent_iter = read_file(trn_file)
		for sent in sent_iter:
			sent_feat_vec, labels = self.get_sent_feat_vec(sent)
			all_feat_vec.append(sent_feat_vec)
			all_labels.append(labels)
			
		print('--Training Perceptron for Parser--')
		self.train_t(all_feat_vec, all_labels, iter_num)
		
	def predict(self, sentence):
		stack, arcs, = [[-1, '_ROOT_', '_ROOT_', None, None]], []
		queue = [[idx, item[0], item[1], None, None] for idx, item in enumerate(sentence)]
		
		while queue != []:
			candidate_labels = []
			if stack != [] and stack[-1][self.INDEX] != -1 and stack[-1][self.INDEX] not in [arc[1] for arc in arcs]:
				candidate_labels.append('LA')
			if stack != [] and queue[0][self.INDEX] not in [arc[1] for arc in arcs]:
				candidate_labels.append('RA')
			if len(queue) != 1 or stack == []:
				candidate_labels.append('SH')
			
			feat_vec = self.extract_feature(stack, queue)
			predict_label = self.predict_t(feat_vec, candidate_labels)
			
			if predict_label == 'LA':
				arcs.append((queue[0][self.INDEX], stack[-1][self.INDEX]))
				queue[0][self.LEFTNODE] = stack[-1]
				stack.pop()
			elif predict_label == 'RA':
				arcs.append((stack[-1][self.INDEX], queue[0][self.INDEX]))
				stack[-1][self.RIGHTNODE] = queue[0]
				queue[0] = stack.pop()
			else:
				stack.append(queue[0])
				del queue[0]
				
		arcs.sort(key = lambda arc: arc[1])
		return [[item[0], item[1], arc[0]] for item, arc in zip(sentence, arcs)]
		
	def predict_and_save(self, infile, outfile):
		sent_iter = read_file(infile)
		handle = open(outfile, 'w', encoding='utf-8')
		for sent in sent_iter:
			predict_sentence = self.predict(sent)
			for item in predict_sentence:
				handle.write('{0}\t{1}\t{2}\n'.format(item[0], item[1], item[2]))
			handle.write('\n')
		handle.close()
		
if __name__ == '__main__':
	parser = Parser()
	parser.train('d:\\Project1\\data\\trn.ec', 15)
	parser.predict_and_save('d:\\Project1\\data\\dev.ec', 'd:\\Project1\\result_4.ec')