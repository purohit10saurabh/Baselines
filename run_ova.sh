# To run: ./run_ova.sh EURLex-4K 0.55 1.5

train (){
    # $1 : dataset
    # $2 : data_dir
    # $3 : model_dir
    # $4 : clf_type
    # $5 : train feat file name
    # $6 : train label file name
    # $7 : optional params
    python3 -W ignore ./ova/main.py -mode 'train' \
    --dataset $1 \
    -clf_type $4 \
    --data_dir $2 \
    --model_dir $3 \
    --tr_feat_fname $5 \
    --tr_label_fname $6 \
    --num_threads 50 \
    --max_iter 50 \
    --norm 'l2' \
    --threshold 0.01 \
    --batch_size 500 \
    --dual True \
    $7
}

predict () {
    # $1: dataset
    # $2: data_dir
    # $3: model_dir
    # $4: result_dir
    # $5: clf_type
    # $6: feature file name
    # $7: label file name
    # $8: opt_params
    python3 -W ignore ./ova/main.py -mode 'predict' \
    --dataset $1 \
    --data_dir "${2}" \
    --model_dir "${3}" \
    -clf_type $5 \
    --norm 'l2' \
    --ts_feat_fname $6 \
    --ts_label_fname $7 \
    --result_dir "${4}" \
    --num_threads 4 \
    --batch_size 1000 \
    $8
}

dataset=$1
data_version=$2
work_dir="$3"
method=$4

data_dir="${work_dir}/data/${data_version}/${dataset}"

model_dir="${work_dir}/models/${data_version}/${dataset}/$method"
result_dir="${work_dir}/results/${data_version}/$dataset/$method"
score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict.txt"

mkdir -p $model_dir
mkdir -p $result_dir

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"


train $dataset $data_dir $model_dir $method $trn_ft_file $trn_lbl_file | tee -a $log_train
predict $dataset "${data_dir}" "${model_dir}" "${result_dir}" $method $tst_ft_file $tst_lbl_file | tee -a "${log_predict}"
