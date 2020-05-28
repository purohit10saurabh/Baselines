import sys
import xclib.data.data_utils as data_utils
from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np

def get_mat(file_name):
	if file_name.endswith('.npz'):
		return sp.load_npz(file_name)
	if file_name.endswith('.txt'):
		return data_utils.read_sparse_file(file_name, safe_read=False)

if __name__ == '__main__':
	pos_file = sys.argv[1]
	predictions_file = sys.argv[2]
	score_dir = sys.argv[3]
	#print(pos_file, predictions_file,score_dir)
	predicted_labels = get_mat(predictions_file)    
	pos_trn = data_utils.read_sparse_file(pos_file, safe_read=False)
	
	pos_score = (predicted_labels>0).astype(np.float32)
	pos_trn = (pos_trn>0).astype(np.float32)
	pos_score = pos_score - pos_trn
	pos_score = pos_score>0
	ans = (predicted_labels.multiply(pos_score)).astype(np.float32)
	ans.eliminate_zeros()
	data_utils.write_sparse_file(ans, score_dir+'/nd_score.txt')	