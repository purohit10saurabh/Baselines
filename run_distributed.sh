#!/bin/bash

dataset=$1
data_version=$2

prefix=$(echo $dataset | tr '-' "d")

stats=`echo $dataset | python3 -c "import sys, json; print(json.load(open('data_stats.json'))['${dataset}'])"` 
stats=($(echo $stats | tr ',' "\n"))

vocabulary_dims=${stats[0]}
num_labels=${stats[1]}

A=${stats[3]}
B=${stats[4]}

echo $dataset $A $B

work_dir="${HOME}/scratch/lab/Workspace"
compile_again=True
mkdir -p "${work_dir}/data/${data_version}/${dataset}"

if [ $data_version == 'dense' ]
then
    if [ ! -e "${work_dir}/data/${data_version}/${dataset}/train.txt" ]
    then
        sh gen_dense_features.sh "${work_dir}/data/${dataset}/train.txt" "${work_dir}/data/${dataset}/fasttextB_embeddings_300d.npy" "wt_avg" "${work_dir}/data/${data_version}/${dataset}/train.txt"
        sh gen_dense_features.sh "${work_dir}/data/${dataset}/test.txt" "${work_dir}/data/${dataset}/fasttextB_embeddings_300d.npy" "wt_avg" "${work_dir}/data/${data_version}/${dataset}/test.txt"
    fi
else
    ln -s "${work_dir}/data/${dataset}/train.txt" "${work_dir}/data/${data_version}/${dataset}/train.txt"
    ln -s "${work_dir}/data/${dataset}/test.txt" "${work_dir}/data/${data_version}/${dataset}/test.txt"
fi
wait

echo "Running 1vsALL"
cd ./SVM-Dense-Parallel
sh distributed_multijobs.sh $dataset $data_version $work_dir
cd $cwd
