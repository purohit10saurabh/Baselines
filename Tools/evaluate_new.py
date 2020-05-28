# Example to evaluate
import sys
import xc_metrics_new as xc_metrics
import xclib.data.data_utils as data_utils
from scipy.io import loadmat
import scipy.sparse as sp
import numpy as np
import csv
#from utils import *

def get_mat(file_name):
    if file_name.endswith('.npz'):
        return sp.load_npz(file_name)
    if file_name.endswith('.txt'):
        return data_utils.read_sparse_file(file_name, safe_read=False)

def dedup_main(targets_file, train_file, predictions_file, A, B, file, pos_file):
    predicted_labels = get_mat(predictions_file) 
    pos_trn = data_utils.read_sparse_file(pos_file, safe_read=False)    
    pos_tst = (predicted_labels>0).astype(np.float32)
    pos_tst = pos_tst - pos_trn
    pos_tst = pos_tst>0
    predicted_labels = predicted_labels.multiply(pos_tst)
    predicted_labels.eliminate_zeros()  

    true_labels = data_utils.read_sparse_file(targets_file, safe_read=False)
    trn_labels = data_utils.read_sparse_file(train_file, safe_read=False)
    inv_propen = xc_metrics.compute_inv_propesity((trn_labels>0).astype(np.float64), A, B)    
    xc_metrics.do_metrics(trn_labels, true_labels, predicted_labels, inv_propen, file)
    
def main(targets_file, train_file, predictions_file, A, B, file):
    # Load the dataset    
    #print(targets_file, "\n",train_file, "\n",predictions_file)
    true_labels = data_utils.read_sparse_file(targets_file, safe_read=False)
    trn_labels = data_utils.read_sparse_file(train_file, safe_read=False)
    predicted_labels = get_mat(predictions_file)    
    #print(true_labels.shape, trn_labels.shape, predicted_labels.shape)
    inv_propen = xc_metrics.compute_inv_propesity((trn_labels>0).astype(np.float64), A, B)

    #print(trn_labels.shape, true_labels.shape, predicted_labels.shape)
    xc_metrics.do_metrics(trn_labels, true_labels, predicted_labels, inv_propen, file)

    #acc = xc_metrics.Metrices(true_labels, inv_propensity_scores=inv_propen, remove_invalid=False, batch_size=50000)
    #args = acc.eval(predicted_labels, 5)
    #print(xc_metrics.format(*args))

if __name__ == '__main__':
    train_file = sys.argv[1]
    targets_file = sys.argv[2] # Usually test data file
    predictions_file = sys.argv[3] # In mat format
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    file = sys.argv[6]
    if "-dedup" in sys.argv:
        d_id = sys.argv.index("-dedup") + 1
        pos_file = sys.argv[d_id]
        dedup_main(targets_file, train_file, predictions_file, A, B, file, pos_file);
    else:
        main(targets_file, train_file, predictions_file, A, B, file)
