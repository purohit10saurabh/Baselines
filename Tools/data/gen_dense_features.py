# Run using python gen_avg_feat.py <data_file> <embeddings> <method> <out_file>
import numpy as np
import sys
import xclib.data.data_utils as data_utils
import numpy as np
import os


def compute_features(features, word_embeddings, method='wt_sum'):
    """
        Compute dense features as per given sparse features and word embeddings
        Args:
            features: csr_matrix: sparse features
            word_embeddings: np.ndarray: dense embedding for each word in vocabulary
            method: str: wt_sum or wt_avg
        Returns:
            document_embeddings: np.ndarray: dense embedding for each document
    """
    print(features.shape, word_embeddings.shape)
    document_embeddings = features.dot(word_embeddings)
    return document_embeddings

def main():
    data_dir = sys.argv[1]
    embeddings_fname = sys.argv[2]
    method = sys.argv[3]
    mode = sys.argv[4]
    feat = data_utils.read_sparse_file(os.path.join(data_dir, '%s_X_Xf.txt'%(mode)), force_header=True)
    labels = data_utils.read_sparse_file(os.path.join(data_dir, '%s_X_Y.txt'%(mode)), force_header=True)
    num_samples= feat.shape[0]
    if mode == 'trn':
        print("Eliminating zeros")
        ft_valid_idx = np.where(np.ravel(feat.sum(axis=1))>0)
        lb_valid_idx = np.where(np.ravel(labels.sum(axis=1)) > 0)
        valid_idx = np.intersect1d(ft_valid_idx,lb_valid_idx)
        print("Total valid instances are %d/%d"%(valid_idx.size,num_samples))
        feat = feat[valid_idx,:]
        labels = labels[valid_idx,:].tolil().tocsr()
        data_utils.write_sparse_file(labels, os.path.join(data_dir, '%s_X_Y.txt'%(mode)))
    word_embeddings = np.load(embeddings_fname).astype(np.float32)
    embedding_dim = word_embeddings.shape[1]
    document_embeddings = compute_features(feat, word_embeddings, method)
    with open(os.path.join(data_dir, '%s_X_Xf.txt'%(mode)),'w+') as fts:
        fts.write("%d %d\n"%document_embeddings.shape)
        np.savetxt(fts, document_embeddings, delimiter=' ', fmt="%f")


if __name__ == '__main__':
    main()
