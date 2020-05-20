from xclib.data import data_utils as du
import sys
import scipy.sparse as sp
import numpy as np


def _remove():
    ft, lb, _, _, _ = du.read_data(sys.argv[2])
    print(ft.shape)
    valid_instances = np.where(np.ravel(ft.sum(axis=1)) > 0)
    ft = ft[valid_instances]
    lb = lb[valid_instances]
    du.write_data(sys.argv[3], features=ft.tocsr(), labels=lb.tocsr())
    np.savetxt(sys.argv[4], valid_instances, fmt="%d")
    print(ft.shape)


def _remapped():
    valid_instances = np.loadtxt(sys.argv[2])
    scores = du.read_sparse_file(sys.argv[3], safe_read=False).tocsc()
    print(scores.shape, valid_instances.shape)
    print(np.max(scores.indices))
    data = scores.data
    indices = np.asarray(list(map(lambda x: valid_instances[x], scores.indices)))
    indptr =  scores.indptr
    instances = int(sys.argv[4])
    _scores = sp.csc_matrix((data, indices, indptr), shape=(instances, scores.shape[1]) ,dtype=np.float32)
    print(_scores.shape)
    du.write_sparse_file(_scores.tocsr(), sys.argv[3])


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == "remove":
        _remove()

    if mode == "remap":
        _remapped()
