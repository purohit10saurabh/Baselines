#!/usr/bin/env bash

DATASET_NAME="EURLex-4K"
FILES_PREFIX="eurlex"
PARAMS="-lr 0.5 -epoch 20 -arity 2 -dim 250 -l2 0.001 -wordsWeights -treeType kmeans"

bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
