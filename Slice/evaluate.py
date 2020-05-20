# Example to evaluate
import sys
import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils
import scipy.sparse as sp
from xclib.utils import sparse
import numpy as np
import _pickle as pickle
import copy
import os


def main(targets_file, train_file, predictions_file, A, B):
    trn_labels = data_utils.read_sparse_file(train_file)
    trn_labels = sparse.binarize(trn_labels)
    true_labels = data_utils.read_sparse_file(targets_file)
    true_labels = sparse.binarize(true_labels)
    inv_propen = xc_metrics.compute_inv_propesity(trn_labels, A, B)
    acc = xc_metrics.Metrices(true_labels, inv_propensity_scores=inv_propen, remove_invalid=False)
    predicted_labels = data_utils.read_sparse_file(predictions_file)
    args = acc.eval(predicted_labels, 5)
    print(xc_metrics.format(*args).replace(" ", ","))


if __name__ == '__main__':
    train_file = sys.argv[1]
    targets_file = sys.argv[2] 
    predictions_file = sys.argv[3]  
    A = float(sys.argv[4])
    B = float(sys.argv[5])
    main(targets_file, train_file, predictions_file, A, B)
