#!/bin/bash

work_dir=$1
dataset=$2
data_version=$3
num_trees=$4

#module load compiler/gcc/4.9.3/compilervars

data_dir="${work_dir}/data/${data_version}/${dataset}"
model_dir="${work_dir}/models/${data_version}/${dataset}/FastXML"
result_dir="${work_dir}/results/${data_version}/${dataset}/FastXML"
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
#trn_ft_lbl_file="${data_dir}/trn_X_XY.txt"
score_file="${result_dir}/score.txt"

mkdir -p $model_dir $result_dir

cd FastXML
echo "Training.."
#echo "$trn_ft_file $trn_lbl_file $model_dir" 
./fastXML_train $trn_ft_file $trn_lbl_file $model_dir -T 5 -s 0 -t 50 \
        -b 1.0 -c 1.0 -m 10 -l 10 &> $log_train

echo "Predicting.."
./fastXML_predict $tst_ft_file $score_file $model_dir &> $log_predict
cd -
