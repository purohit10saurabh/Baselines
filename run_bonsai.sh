#!/bin/bash

work_dir=$1
dataset=$2
data_version=$3
num_trees=$4

module load compiler/gcc/4.9.3/compilervars

data_dir="${work_dir}/data/${data_version}/${dataset}"
model_dir="${work_dir}/models/${data_version}/${dataset}/Bonsai"
result_dir="${work_dir}/results/${data_version}/${dataset}/Bonsai"
score_file="${result_dir}/score.txt"
log_train="${result_dir}/log_train.txt"
log_predict="${result_dir}/log_predict"
log_evaluate="${result_dir}/log_evaluate.txt"

mkdir -p $model_dir
mkdir -p $result_dir

trn_ft_file="${data_dir}/trn_X_Xf.txt"
trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_ft_file="${data_dir}/tst_X_Xf.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"
trn_ft_lbl_file="${data_dir}/trn_X_XY.txt"
score_file="${result_dir}/score.txt"

mkdir -p $model_dir $result_dir

cd bonsai/shallow
python3 label_append_train.py $data_dir
echo "Training.."
./bonsai_train $trn_ft_file $trn_lbl_file $trn_ft_lbl_file $model_dir \
    -T ${num_trees} -s 0 -t 3 -w 100 -b 1.0 -c 1.0 -m 3 -f 0.1 -fcent 0 \
    -k 0.0001 -siter 20 -q 0 -ptype 0 -ctype 0 > $log_train
predict(){
    score_mat="${data_dir}/score_0.txt"
    tst_file="${data_dir}/tst_X_Xf_0.txt"
    ./bonsai_predict $tst_file $score_mat $model_dir \
        -T ${num_trees} -s 0 -t 3 -B 10 -d 0.98 -q 0 >"${log_predict}_0.txt" &
    k=1
    for ((lr_idx = 1; lr_idx < $total_batches; lr_idx++)); do
        score_mat="${data_dir}/score_$lr_idx.txt"
        tst_file="${data_dir}/tst_X_Xf_$lr_idx.txt"
        ./bonsai_predict $tst_file $score_mat $model_dir \
            -T ${num_trees} -s 0 -t 3 -B 10 -d 0.98 -q 0 >"${log_predict}_$lr_idx.txt" &
        (( k++ ))
        if [ $k -eq 5 ]; then
            k=0
            wait
            echo "Done $lr_idx / $total_batches"
        fi
    done
    wait
    echo "Done $lr_idx / $total_batches"
}

echo "Predicting.."
total_batches=$(python -u -W ignore batches.py $tst_ft_file 1000 "write")
wait
echo $total_batches
predict
tot_pred_time="0"
for ((lr_idx = 0; lr_idx < $total_batches; lr_idx++)); do
    pred_file="${log_predict}_${lr_idx}.txt"
    tail -n 2 ${pred_file} > "$data_dir/dummy.txt"
    pred_time=$( head -n 1 $data_dir/dummy.txt | awk -F ' ' '{printf "%0.2f",$3}')
    tot_pred_time="$( printf '%f + %f\n' "$pred_time" "$tot_pred_time" | bc )"
done
tot_pred_time="$( printf 'scale=2; %f / %f\n' "$tot_pred_time" "$total_batches" | bc )"
model_size=$(tail -1 ${log_predict}_0.txt | awk -F ' ' '{printf "%0.2f",$3}')

echo $tot_pred_time > "${log_predict}.txt"
echo $model_size >> "${log_predict}.txt"
score_mat="${data_dir}/score.txt"
python -u -W ignore batches.py $score_mat $total_batches "concat"
mv $score_mat $score_file
cd -
