work_dir="${HOME}/scratch/XC"
data_dir="${work_dir}/data"
result_dir="${work_dir}/results"
raw_dir="${work_dir}/RawData"
dataset=$1
raw_data=$2
A=$3
B=$4
data_version="sparse"

convert () {
    data_tools="./tools/data"
	perl $data_tools/convert_format.pl $1 $2 $3
	perl $data_tools/convert_format.pl $4 $5 $6
}


run_on_dataset(){
    temp_dataset=$1
    num_trees=3
    train_file="${model_data_dir}/train.txt"
    test_file="${model_data_dir}/test.txt"
    trn_ft_file="${model_data_dir}/trn_X_Xf.txt"
    trn_lbl_file="${model_data_dir}/trn_X_Y.txt"
    tst_ft_file="${model_data_dir}/tst_X_Xf.txt"
    tst_lbl_file="${model_data_dir}/tst_X_Y.txt"
    convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}
    ./run_parabel_old.sh $work_dir $temp_dataset $data_version $num_trees
}


run_mach(){
    temp_dataset=$1
    ./run_deeplearning.sh 0 titles ${temp_dataset} ${raw_data} $A $B MACH filter_labels.txt 2000
}

evaluate(){
    temp_dataset=$1
    trn_lbl_file="${model_data_dir}/trn_X_Y.txt"
    tst_lbl_file="${model_data_dir}/tst_X_Y.txt"
    echo $score_file
    args="filter_labels.txt \
          ${model_data_dir} \
          ${raw_dir}/${raw_data}
          ${tst_lbl_file} \
          ${tst_lbl_file} \
          $A $B 1"
    python3 -u -W ignore Tools/python/evaluate.py ${args} ${score_file}
}

# alphas=("0.9" "0.3" "0.1" "0.0")
alphas=("0.5" "1.0" "0.9" "0.8" "0.3" "0.0")
# mkdir -p "${data_dir}/${dataset}/warm_data"

for alpha in "${alphas[@]}"
do  
    echo "alpha=", $alpha
    lbl_fts="${data_dir}/${dataset}/Yfts-corpus/Yf.txt"
    model_data_dir="${data_dir}/${data_version}/${dataset}_alpha_${alpha}"
    mkdir -p "${model_data_dir}"
    args="${data_dir}/$dataset ${model_data_dir} ${alpha} ${lbl_fts}"
    run_mach "${dataset}_alpha_${alpha}"

    # python3 Tools/python/warm_start_features.py ${args}
    # run_on_dataset "${dataset}_alpha_${alpha}"
    # cp "${data_dir}/$dataset/filter_labels_test.txt" "${model_data_dir}/filter_labels.txt"
    # score_file="${result_dir}/${data_version}/${temp_dataset}/ParabelOld/score.txt"
    # evaluate "${dataset}_alpha_${alpha}"
done
# rm -rf "${data_dir}/${data_version}"
