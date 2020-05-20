#!/bin/bash
# $1: docoument in sparse xc format
# $2: word embeddings in numpy format
# $3: method (wt_sum or wt_avg)
# $4: output file
# Example: ./gen_dense_features.sh ../data/dense/EURLex-4K/train_sparse.txt fasttextB_embeddings_300d.npy ../data/dense/EURLex-4K/train.txt

temp_file=$(mktemp /tmp/foo.XXXXXXXXX)

python3 ./Tools/data/gen_dense_features.py $1 $2 $3 $4 $5