#!/bin/bash
# Create baselines for different embeddings

compile () {
    make
}

train_parabel () {
    # $1: train feature file name
    # $2: train label file name
    # $3: model directory
    # $4: start tree number
    # $5: number of trees
	./parabel_train $3 $4 --trn_ft_file $1 --trn_lbl_file $2 -T $5 -s 0 -t $5 -c 1.0 -m 100 -k 0
    # ./parabel_train $1 $2 $3 -T $5 -s 0 -t $5
}

predict_parabel () {
    # $1: feature file name
    # $2: score file name
    # $3: model directory
    # $4: start tree number
    ./parabel_predict $3 $4 $2 --tst_ft_file $1
    # ./parabel_predict $1 $3 $2
}


run_parabel () {
    # $1 train feature file
    # $2 train label file
    # $3 test feture file
    # $4 test label file
    # $5 num trees
    # $6 model directory
    # $7 score file
    # $8 log train file
    # $9 log predict
    # $10 log evaluate
    cd Parabel
    echo "Training.."
    train_parabel $1 $2 $6 0 $5 > $8
    echo "Predicting.."
    predict_parabel $3 $7 $6 0 >> $9
    cd -
}


work_dir=$1
dataset=$2
data_version=$3
num_trees=$4

data_dir="${work_dir}/data/${data_version}/${dataset}"

model_dir="${work_dir}/models/${data_version}/${dataset}/Parabel"
result_dir="${work_dir}/results/${data_version}/${dataset}/Parabel"
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

run_parabel $trn_ft_file $trn_lbl_file $tst_ft_file $tst_lbl_file $num_trees $model_dir $score_file $log_train $log_predict $log_evaluate

