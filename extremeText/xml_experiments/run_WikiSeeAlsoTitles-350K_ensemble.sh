#!/usr/bin/env bash

DATASET_NAME="WikiSeeAlsoTitles-350K"
FILES_PREFIX="WikiSeeAlsoTitles350K"
PARAMS="-lr 0.1 -epoch 30 -arity 2 -dim 500 -l2 0.002 -wordsWeights -treeType kmeans -ensemble 3"

# bash run_xml.sh $DATASET_NAME $FILES_PREFIX "$PARAMS"
