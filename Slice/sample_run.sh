#!/bin/bash
set -e

dataset="EURLex-4K"
data_dir="./Sandbox/Data/$dataset"
results_dir="./Sandbox/Results/$dataset"
model_dir="./Sandbox/Results/$dataset/model"
mkdir -p $model_dir

trn_ft_file="${data_dir}/xmlcnn_trn_ft_mat_dense.txt"
trn_lbl_file="${data_dir}/xmlcnn_trn_lbl_mat.txt"
tst_ft_file="${data_dir}/xmlcnn_tst_ft_mat_dense.txt"
tst_lbl_file="${data_dir}/xmlcnn_tst_lbl_mat.txt"
score_file="${results_dir}/score_mat.txt"


#echo "Converting sparse feature matrices to dense format"
#./Tools/c++/smat_to_dmat [sparse train feature file] $trn_ft_file
#./Tools/c++/smat_to_dmat [sparse test feature file] $tst_ft_file


echo "----------------Slice--------------------------"
./slice_train $trn_ft_file $trn_lbl_file $model_dir -m 100 -c 300 -s 300 -k 300 -o 20 -t 1 -f 0.000001 -siter 20 -b 2 -stype 0 -C 1 -q 0
./slice_predict $tst_ft_file $model_dir $score_file
./Tools/metrics/precision_k $score_file $tst_lbl_file 5
./Tools/metrics/nDCG_k $score_file $tst_lbl_file 5

