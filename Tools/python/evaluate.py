import numpy as np
from xclib.data import data_utils as du
from xclib.evaluation import xc_metrics as xc
import scipy.sparse as sp
import sys
import os

filter_labels = sys.argv[1]
data_dir = sys.argv[2]
test, train = sys.argv[3], sys.argv[4]
A = float(sys.argv[5])
B = float(sys.argv[6])
filter_data = int(sys.argv[7])
methods = sys.argv[8:]

def load_overlap(data_dir, sep='->', key='test',
                 filter_label_file='filter_labels.txt'):
    if os.path.exists(os.path.join(data_dir, filter_label_file)):
        filter_lbs = np.loadtxt(os.path.join(
            data_dir, filter_label_file), dtype=np.int32)
        docs = filter_lbs[:, 0]
        lbs = filter_lbs[:, 1]
    else:
        
        lbs, docs = [], []
        lbs, docs = np.array(lbs, dtype=np.int32), np.array(
            docs, dtype=np.int32)
        print("(Overlap file not found)")
    return docs, lbs


def get_mat(file_name):
    if file_name.endswith('.npz'):
        return sp.load_npz(file_name)
    if file_name.endswith('.txt'):
        return du.read_sparse_file(file_name, safe_read=False)


def _remove_overlap(score_mat, docs, lbs):
    score_mat[docs, lbs] = 0
    score_mat.eliminate_zeros()
    return score_mat

Y = get_mat(test)
TY = get_mat(train)
inv_prop = xc.compute_inv_propesity(labels=TY, A=A, B=B)
docs, lbs = load_overlap(data_dir, filter_label_file=filter_labels)
Y = _remove_overlap(Y, docs, lbs)
acc = xc.Metrices(true_labels=Y, inv_propensity_scores=inv_prop)
for mat in methods:
    method = mat.split('/')[-2]
    score_mat = get_mat(mat)
    #print(Y.shape, TY.shape, score_mat.shape)
    if filter_data:
        score_mat = _remove_overlap(score_mat, docs, lbs)
    for accuracy in acc.eval(score_mat):
        print(xc.format(accuracy[0::2]), end=',')
    print()
    del score_mat
