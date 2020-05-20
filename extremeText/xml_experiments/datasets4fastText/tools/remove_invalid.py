from xclib.data import data_utils as du
import scipy.sparse as sp
import numpy as np
import sys

def remove_invalid(file, output):
    print(file)
    ft, lb, examples, _, _ = du.read_data(file, header=True)
    valid_doc_featrues = np.where( np.ravel(ft.sum(axis=1))>0)[0]
    valid_doc_labels = np.where( np.ravel(lb.sum(axis=1))>0)[0]
    valid_doc = np.intersect1d(valid_doc_featrues, valid_doc_labels)
    ft = ft[valid_doc].tolil()
    lb = lb[valid_doc].tolil()
    du.write_data(output, features=ft.tocsr(), labels=lb.tocsr())
    print("Total valid docs {}/{}".format(valid_doc.size, examples))
    np.savetxt(output+'.ids', valid_doc, fmt="%d")

def remap_score_ids(file, root_ids):
    pass
    # print("Remapping to original index")
    # score = du.read_sparse_file(file, safe_read=False, dtype=np.float32)
    # num_instances = int(open(root_ids+'.examples', 'r').readline().strip())
    # _score = sp.lil_matrix((num_instances, score.shape[1]), dtype=np.float32)
    # valid_docs = np.loadtxt(root_ids+'.ids', dtype=int)
    # _score[valid_docs] = score
    # _score = _score.tolil()
    # print("Remapped to original index")
    # du.write_sparse_file(_score.tocsr(), file)

if __name__=='__main__':
    flag = sys.argv[1]
    if flag == 'remove':
        remove_invalid(sys.argv[2], sys.argv[3])
    elif flag == 'remap':
        remap_score_ids(sys.argv[2], sys.argv[3])
