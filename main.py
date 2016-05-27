from ECDetection import ECDetection
from Parser import Parser
from DepLabel import DepLabel

if __name__ == '__main__':
	dectection = ECDetection()
	dectection.train('data\\trn.ec')
	dectection.predict_and_save('data\\tst.ec', 'tmp_ec')
	
	parser = Parser()
	parser.train('data\\trn.ec')
	parser.predict_and_save('tmp_ec', 'tmp2_ec')
	
	depLabel = DepLabel()
	depLabel.train('data\\trn.ec')
	depLabel.predict_and_save('tmp2_ec', 'tst_ans.ec')
	