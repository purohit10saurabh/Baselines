import data_utils
import sys
from scipy.sparse import hstack, csr_matrix
import numpy as np

input = sys.argv[1]
output = sys.argv[2]

features, labels, num_samples, num_feat, num_labels = data_utils.read_data(input, header=False)
labels = data_utils.binarize_labels(labels, num_labels+1)

features = data_utils.normalize_data(features)

data_utils.write_data(output, hstack([csr_matrix((num_samples, 1), dtype=np.float32), features]).tocsr(), labels)