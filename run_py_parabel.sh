#!/bin/bash
# Create baselines for different embeddings

work_dir="${HOME}/scratch/lab/Workspace"
data_dir="${work_dir}/data"
dset=$1
model_dir="${work_dir}/models/sparse/${dset}/Parabel"
result_dir="${work_dir}/results/sparse/${dset}/Parabel"

mkdir -p $model_dir
mkdir -p $result_dir

python3 parabel.py "${data_dir}/${dset}/train.txt" "${data_dir}/${dset}/test.txt" $model_dir 3 $result_dir 'train'
python3 parabel.py "${data_dir}/${dset}/train.txt" "${data_dir}/${dset}/test.txt" $model_dir 3 $result_dir 'predict'


fetch_p_data_rest(){
    train_file=$trn_lbl_file
    test_file=$tst_lbl_file
    score_file="${score_dir}/${METHOD}/${FILE}"
    A=$A
    B=$B
    echo -ne $(python3 -W ignore tools/evaluate.py $train_file $test_file $score_file $A $B |awk -F ',' '{printf "%0.2f,%0.2f,%0.2f,",$1,$3,$5}')
    echo -ne $(tail -2 ${work_dir}/results/sparse/${dset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$5}')
    echo $(tail -1 ${work_dir}/results/sparse/${dset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$7}')
}

echo "Algo P1 P3 P5 N1 N3 N5 PSP1 PSP3 PSP5 PSN1 PSN3 PSN5 MODELSIZE TRNTIME PREDTIME" |tee "${work_dir}/results/sparse/${dset}/summary.txt"
train_file="${work_dir}/data/${dset}/train.txt"
test_file="${work_dir}/data/${dset}/test.txt"
score_dir="${work_dir}/results/sparse/${dset}"
FILE='score.txt'

METHOD='Parabel'
echo $METHOD $(fetch_p_data_rest) | tee -a "${work_dir}/results/sparse/${dset}/summary.txt"
