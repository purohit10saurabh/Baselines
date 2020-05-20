from xclib.data import data_utils as du
import sys

if __name__=='__main__':
    score_mat = du.read_sparse_file(sys.argv[2])
    for files in sys.argv[3:]:
        print(files)
        score_mat = score_mat + du.read_sparse_file(files)

    du.write_sparse_file(X = score_mat, filename=sys.argv[1])