#!/usr/bin/env bash

DATASET_NAME="WikiTitles-500K"
FILES_PREFIX="WikiTitles500K"
PARAMS="-lr 0.05 -epoch 30 -arity 2 -dim 500 -l2 0.001 -wordsWeights -treeType kmeans -ensemble 3"

# bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
