#!/usr/bin/env bash

DATASET_NAME="AmazonCat-13K"
FILES_PREFIX="amazonCat"
PARAMS="-lr 0.1 -epoch 20 -arity 2 -dim 500 -l2 0.001 -wordsWeights -treeType kmeans -ensemble 3"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
