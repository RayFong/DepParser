import time
from collections import defaultdict

class Perceptron:
	def __init__(self, label, feature):
		self.label_names, self.feature_names = label, feature  # 类别、特征名称
		self.cur = 0
		
		self.weight, self.total, self.tstamps = {}, {}, {}
		for feature in self.feature_names:
			self.weight[feature] = {}
			self.total[feature] = {}
			self.tstamps[feature] = {}
			for label in self.label_names:
				self.weight[feature][label] = defaultdict(int)
				self.total[feature][label] = defaultdict(int)
				self.tstamps[feature][label] = defaultdict(int)
		
	def update(self, feat_vec, correct_label, predict_label):
		for feat_name in self.feature_names:
			feat_val = feat_vec[feat_name] # 特征向量中该特征的值
			if feat_val != '_NULL_':
				tstamps1 = self.tstamps[feat_name][correct_label][feat_val]
				tstamps2 = self.tstamps[feat_name][predict_label][feat_val]
				self.total[feat_name][correct_label][feat_val] += self.weight[feat_name][correct_label][feat_val] * (self.cur - tstamps1)
				self.total[feat_name][predict_label][feat_val] += self.weight[feat_name][predict_label][feat_val] * (self.cur - tstamps2)
				self.tstamps[feat_name][correct_label][feat_val] = self.cur
				self.tstamps[feat_name][predict_label][feat_val] = self.cur
				
				# 更新权重
				self.weight[feat_name][correct_label][feat_val] += 1
				self.weight[feat_name][predict_label][feat_val] -= 1
				
	def predict_t(self, feat_vec, candidate_label):
		scores = defaultdict(int) # 计算每个类别的得分
		for feat_name in self.feature_names:
			feat_val = feat_vec[feat_name]
			for label in candidate_label:
				scores[label] += self.weight[feat_name][label][feat_val]
				
		predict_label = max(candidate_label, key = lambda label : (scores[label], label)) # 选取得分最高的作为预测值
		return predict_label
		
	def train_t(self, all_feat_vec, all_labels, iter_num = 10):
		for iter in range(iter_num):
			time.clock()
			iteration_sentence_accuracy, iteration_word_accuracy = [0, 0], [0, 0]
			for (sent_feat_vec, sent_labels) in zip(all_feat_vec, all_labels):
				sent_correct = True
				for (feat_vec, correct_label) in zip(sent_feat_vec, sent_labels):
					predict_label = self.predict_t(feat_vec, self.label_names)
					if predict_label == correct_label:
						iteration_word_accuracy[0] += 1
					else: 	# 预测错误，修改权重
						self.update(feat_vec, correct_label, predict_label)
						sent_correct = False
					self.cur += 1
					iteration_word_accuracy[1] += 1
				iteration_sentence_accuracy[0] += (sent_correct == True)
				iteration_sentence_accuracy[1] += 1
				
			# result
			sentence_accuracy = iteration_sentence_accuracy[0] / iteration_sentence_accuracy[1];
			word_accuracy = iteration_word_accuracy[0] / iteration_word_accuracy[1];
			print('iteration {0} at time {1}'.format(iter+1, str(time.clock())))
			print('  sentence accuracy: {0}, ({1}/{2})'.format(sentence_accuracy, \
				iteration_sentence_accuracy[0], iteration_sentence_accuracy[1]))
			print('  word accuracy: {0}, ({1}/{2})'.format(word_accuracy, \
				iteration_word_accuracy[0], iteration_word_accuracy[1]))
				
	def get_average_weight(self):
		for feat_name in self.feature_names:
			for label in self.label_names:
				for feat_val, w in self.weight[feat_name][label].items():
					totals = self.total[feat_name][label][feat_val]
					totals += w * (self.cur - self.tstamps[feat_name][label][feat_val])
					average = round(totals / float(self.cur), 3)
					if average:
						self.weight[feat_name][label][feat_val] = average