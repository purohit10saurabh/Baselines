#!/bin/bash
# Create baselines for different embeddings

train_slice () {
    # $1: train feature file name
    # $2: train label file name
    # $3: model directory
    # $4: M
    # $5: efC
    # $6: efS
    /usr/bin/time -v ./slice_train $1 $2 $3 -m $4 -c $5 -s $6 -k $6 -o 20 -t 12 -C 1 -f 0.000001 -siter 20 -stype 0 -q 0
}

predict_slice () {
    # $1: feature file name
    # $2: score file name
    # $3: model directory
    ./slice_predict $1 $3 $2 -b 0 -t 1 -q 0
}


run_slice () {
    # $1 train feature file
    # $2 train label file
    # $3 test feture file
    # $4 test label file
    # $5 efC
    # $6 efS
    # $7 M
    # $8 model directory
    # $9 score file
    # $10 log train file
    # $11 log predict
    cd Slice
    echo "Training.."
    train_slice $1 $2 $8 $7 $5 $6 | tee ${10}
    echo "Predicting.."
    predict_slice $3 $9 $8 | tee ${11}
    cd -
}

evaluate () {
    # $1 train
    # $2 target
    # $3 prediction
    # $4 A
    # $5 B
    python evaluate.py $1 $2 $3 $4 $5
}

efC=300
M=100
efS=300
dataset=$1
data_version=$2
work_dir="$3"

data_dir="${work_dir}/data/${data_version}/${dataset}"

model_dir="${work_dir}/models/${data_version}/${dataset}/Slice"
result_dir="${work_dir}/results/${data_version}/$dataset/Slice"
score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict.txt"
log_evaluate="${result_dir}/log_evaluate.txt"

mkdir -p $model_dir
mkdir -p $result_dir

trn_ft_file="${data_dir}/tr_X_Xf.txt"
trn_lbl_file="${data_dir}/tr_X_Y.txt"
tst_ft_file="${data_dir}/ts_X_Xf.txt"
tst_lbl_file="${data_dir}/ts_X_Y.txt"

run_slice $trn_ft_file $trn_lbl_file $tst_ft_file $tst_lbl_file $efC $efS $M $model_dir $score_file $log_train $log_predict
