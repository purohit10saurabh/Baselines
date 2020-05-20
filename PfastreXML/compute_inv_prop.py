import sys
from xclib.data import data_utils
from xclib.evaluation.xc_metrics import compute_inv_propesity
import numpy as np


def main():
    train_labels = data_utils.read_sparse_file(sys.argv[1])
    A = float(sys.argv[2])
    B = float(sys.argv[3])
    np.savetxt(sys.argv[4], compute_inv_propesity(train_labels, A, B), fmt='%f')


if __name__ == "__main__":
    main()
