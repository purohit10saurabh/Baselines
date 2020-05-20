from xclib.data import data_utils as du
from sklearn.preprocessing import normalize as scale
import scipy.sparse as sp
import numpy as np
import sys
import os


def correct_dataset(ft, num_fts):
    if ft.shape[1] != num_fts:
        print("Padding fts", num_fts-ft.shape[1])
        ft = sp.hstack(
            [ft, sp.csr_matrix((ft.shape[0], num_fts-ft.shape[1]))]).tocsr()
    return ft


def write_todata(file_name, fts, lbs, lb_fts, alpha):
    # new_fts = sp.hstack([fts.multiply(alpha), lb_fts.multiply(1-alpha)]).tocsr()
    new_fts = fts.multiply(alpha) + lb_fts.multiply(1-alpha)
    du.write_data(file_name, new_fts.tocsr(), lbs)


if __name__ == '__main__':
    data_dir = sys.argv[1]
    out_dir = sys.argv[2]
    alpha = float(sys.argv[3])
    
    trn_fts = du.read_sparse_file(os.path.join(data_dir, 'trn_X_Xf.txt'))
    trn_lbs = du.read_sparse_file(os.path.join(data_dir, 'trn_X_Y.txt'))
    trn_rlb = du.read_sparse_file(os.path.join(data_dir, 'trn_X_RL.txt'))

    tst_fts = du.read_sparse_file(os.path.join(data_dir, 'tst_X_Xf.txt'))
    tst_lbs = du.read_sparse_file(os.path.join(data_dir, 'tst_X_Y.txt'))
    tst_rlb = du.read_sparse_file(os.path.join(data_dir, 'tst_X_RL.txt'))

    
    if not os.path.exists(os.path.join(data_dir, 'warm_data/corp_fts_test_l2.txt')):
        print("Creating data")
        lbl_fts = du.read_sparse_file(sys.argv[4])
        tst_revealed_labels = scale(tst_rlb, norm='l1').dot(lbl_fts)
        trn_revealed_labels = scale(trn_rlb, norm='l1').dot(lbl_fts)
        
        du.write_sparse_file(trn_revealed_labels.tocsr(),
                             os.path.join(data_dir,
                                          'warm_data/corp_fts_train_l2.txt'))
        du.write_sparse_file(tst_revealed_labels.tocsr(),
                             os.path.join(data_dir,
                                          'warm_data/corp_fts_test_l2.txt'))
        
    else:
        trn_revealed_labels = du.read_sparse_file(
            os.path.join(data_dir, 'warm_data/corp_fts_train_l2.txt')
        )[:, :-1].tocsr()
        tst_revealed_labels = du.read_sparse_file(
            os.path.join(data_dir, 'warm_data/corp_fts_test_l2.txt')
        )[:, :-1].tocsr()
    write_todata(os.path.join(out_dir, 'test.txt'),
                 tst_fts, tst_lbs, tst_revealed_labels, alpha)
    write_todata(os.path.join(out_dir, 'train.txt'),
                 trn_fts, trn_lbs, trn_revealed_labels, alpha)
