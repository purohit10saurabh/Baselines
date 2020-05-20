#!/bin/bash

dataset=$1
data_version=$2

prefix=$(echo $dataset | tr '-' "d")

stats=`echo $dataset | python3 -c "import sys, json; print(json.load(open('data_stats.json'))['${dataset}'])"` 
stats=($(echo $stats | tr ',' "\n"))

# vocabulary_dims=${stats[0]}
# num_labels=${stats[1]}

A=${stats[3]}
B=${stats[4]}

echo $dataset $A $B

work_dir="${HOME}/scratch/lab/Workspace"
compile_again=True
mkdir -p "${work_dir}/data/${data_version}/${dataset}"

data="${work_dir}/data/${data_version}/${dataset}/test.txt"
num_test_samples=$(head -n 1 $data |awk -F ' ' '{print $1}')
cwd=$(pwd)
echo $num_labels $num_test_samples


fetch_p_data_rest(){
    train_file=$1
    test_file=$2
    score_file=$3
    A=$4
    B=$5
    echo -ne $(python evaluate.py $train_file $test_file $score_file $A $B |awk -F ',' '{print $1,$3,$5}')
    # echo $out
}

echo "Algo P1 P3 P5 N1 N2 N3 PSP1 PSP3 PSP5 PSN1 PSN3 PSN5" |tee "${work_dir}/results/${data_version}/${dataset}/summary.txt"
train_file="${work_dir}/data/${data_version}/${dataset}/train.txt"
test_file="${work_dir}/data/${data_version}/${dataset}/test.txt"
score_dir="${work_dir}/results/${data_version}/${dataset}"
FILE='score.txt'

echo "Running 1vsALL"
cd ./SVM-Dense-Parallel
sh combine_dmat.sh $dataset $data_version $work_dir
cd $cwd

METHOD='1vsAll'
echo $METHOD $(fetch_p_data_rest $train_file $test_file "${score_dir}/${METHOD}/${FILE}" $A $B) | tee -a "${work_dir}/results/${data_version}/${dataset}/summary.txt"
