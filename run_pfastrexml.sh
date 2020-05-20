#!/bin/bash
# Create baselines for different embeddings

compile () {
    make
}

compute_inv_prop () {
    # $1: Train labels
    # $2: A
    # $3: B
    # $4: out_file
    python3 compute_inv_prop.py $1 $2 $3 $4
}

train_pfastrexml () {
    # $1: train feature file name
    # $2: train label file name
    # $3: model directory
    # $4: inv_prop file
    # $5: number of trees
    ./PfastreXML_train $1 $2 $4 $3 -S 0 -T 5 -s 0 -t $5 -b 1.0 -c 1.0 -m 10 -g 30 -a 0.8
}

predict_pfastrexml () {
    # $1: feature file name
    # $2: score file name
    # $3: model directory
    ./PfastreXML_predict $1 $2 $3
}


run_pfastrexml () {
    # $1 train feature file
    # $2 train label file
    # $3 test feture file
    # $4 test label file
    # $5 num trees
    # $6 model directory
    # $7 score file
    # $8 inv_prop_file
    # $9 log train file
    # $10 log predict
    # $11 log evaluate
    cd PfastreXML
    compute_inv_prop $2 $A $B ${8}
    echo "Training.."
    train_pfastrexml $1 $2 $6 $8 $5 > $9
    echo "Predicting.."
    predict_pfastrexml $3 $7 $6 >> ${10}
    cd -
}


work_dir=$1
dataset=$2
data_version=$3
num_trees=$4
A=$5
B=$6

data_dir="${work_dir}/data/${data_version}/${dataset}"

model_dir="${work_dir}/models/${data_version}/${dataset}/PfastreXML"
result_dir="${work_dir}/results/${data_version}/${dataset}/PfastreXML"
inv_prop_file="${model_dir}/inv_prop.txt"
score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict.txt"
log_evaluate="${result_dir}/log_evaluate.txt"

mkdir -p $model_dir
mkdir -p $result_dir


trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"

run_pfastrexml $trn_ft_file $trn_lbl_file $tst_ft_file $tst_lbl_file $num_trees $model_dir $score_file ${inv_prop_file} $log_train $log_predict $log_evaluate

