from scipy import sparse
import sys
import os
import numpy as np

predicted_lbls = np.loadtxt(sys.argv[1], dtype = "int32", delimiter=",")
predicted_lbls_csr = sparse.dok_matrix((predicted_lbls.shape[0], int(sys.argv[2])), dtype="float32")
labels = int(sys.argv[2])
for i in range(predicted_lbls.shape[0]):
    for j in predicted_lbls[i]:
        if (j > labels):
            continue
        predicted_lbls_csr[i, j - 1] = 1.

sparse.save_npz(os.path.join(os.path.dirname(sys.argv[1]),"score.npz"), predicted_lbls_csr.tocsr())
