from xclib.data import data_utils as du
import sys
import numpy as np
import multiprocessing as mt
import scipy.sparse as sp


def _write_file(data, file, id):
    data = data.tocsr()
    file = file.replace(".txt", "_%d.txt"%(id))
    data.sort_indices()
    du.write_sparse_file(data, file)

def _read_file(file, id):
    file = file.replace(".txt", "_%d.txt"%(id))
    return du.read_sparse_file(file)

def write(file, batch):
    data_obj = du.read_sparse_file(file)
    num_inst = data_obj.shape[0]
    with mt.Pool(5) as p:
        p.starmap(_write_file, iter(map(
            lambda x: (data_obj[x[1]: min(x[1]+batch, num_inst)], file, x[0]),
            enumerate(np.arange(0, num_inst, batch)))))
    print(int(np.ceil(num_inst/batch)))
    return np.ceil(num_inst/batch)

def read(file_suffix, num_objs):
    data = []
    with mt.Pool(5) as p:
        data = p.starmap(_read_file, iter(map(
            lambda x: (file_suffix, x),
            np.arange(num_objs))))
    data = sp.vstack(data).tocsr()
    data.sort_indices()
    du.write_sparse_file(data, file_suffix)

if __name__ == '__main__':
    file = sys.argv[1]
    params = int(sys.argv[2])
    mode = sys.argv[3]
    if mode == "write":
        write(file, params)
    if mode == "concat":
        read(file, params)
