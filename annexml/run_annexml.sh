#!/bin/bash
# Create baselines for different embeddings

train () {
    # $1: train_file
    # $2: model file
    echo $2
    src/annexml train "params.json" train_file=$1 model_file=$2
}

split () {
    # $1: infile
    pred_bak="${1}_backup"
    cp $1 $pred_bak
    cat ${pred_bak} |awk -F '\t' '{print $2}' >$1
    sed -i 's/\,/\ /g' $1
}

convert () {
    # $1: prediction file
    # $2: header
    split $1
    sed -i "1s/^/${2}\n/" $1
}

predict () {
    # $1: test file
    # $2: model file
    # $3: result file
    src/annexml predict "params.json" predict_file=$1 model_file=$2 result_file=$3
}

evaluate () {
    # $1: result file
    cat $1 | python scripts/learning-evaluate_predictions.py
}

compile () {
    make
}

work_dir="/home/kd/Desktop/Workspace"
dataset=$1

#cd annexml

data_dir="${work_dir}/data/${data_version}/${dataset}"
model_dir="${work_dir}/models/AnnexML/${dataset}"
result_dir="${work_dir}/results/AnnexML/${dataset}"

score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict.txt"
log_evaluate="${result_dir}/log_evaluate.txt"


mkdir -p $model_dir
mkdir -p $result_dir

train "${data_dir}/train.txt" "${model_dir}/model" >> $log_train
python3 remove_invalid_remap.py "remove" $TEST $TEST'new' $data_dir'/valid_instances.npy'

test_header=`head -1 "${data_dir}/test.txtnew"`
num_test_samples=`echo $test_header | cut -d' ' -f1`
num_labels=`echo $test_header | cut -d' ' -f3`
test_header="$num_test_samples $num_labels"
predict "${data_dir}/test.txt" "${model_dir}/model" $score_file >> $log_predict
evaluate $score_file >> $log_evaluate
convert $score_file "${test_header}"

test_header=`head -1 "${data_dir}/test.txt"`
num_test_samples=`echo $test_header | cut -d' ' -f1`
python3 remove_invalid_remap.py "remap" $data_dir'/valid_instances.npy' $score_file $num_test_samples

