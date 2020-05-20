from __future__ import print_function
import sys
import numpy as np
from scipy.sparse import csr_matrix
import xclib.evaluation.xc_metrics as xc_metrics

def read_spmat_file(file_path):
    def convert(x):
        l = x.split(":")
        return (int(l[0]), float(l[1]))
    data, rows, cols = [], [], []
    # spmat_vectors = []
    with open(file_path, "r", encoding="utf-8") as fin:
        for i, line in enumerate(fin):
            if(i == 0):
                num_lines, num_Z = list(map(int, line.strip().split()))
                continue
            l = [convert(x) for x in line.strip().split()]
            data += [x[1] for x in l]
            cols += [x[0] for x in l]
            rows += [i -1] * len(l)
    return csr_matrix((data, (rows, cols)), shape=(num_lines, num_Z))

if __name__ == '__main__':
    true_labels_file = sys.argv[1]
    predicted_labels_file = sys.argv[2]
    K = int(sys.argv[3])
    
    print("reading true labels...")
    true_labels = read_spmat_file(true_labels_file)
    print("reading predicted labels...")
    predicted_labels = read_spmat_file(predicted_labels_file)
    
    print(true_labels.shape, predicted_labels.shape)
    print("calculating metrics")
    acc = xc_metrics.Metrices(true_labels)
    acc = acc.eval(predicted_labels, K)
    print(acc)