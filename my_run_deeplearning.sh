#!/bin/bash
#./run_deeplearning titles AmazonTitles-670K Amazon-670K 0.6 2.6 MACH
export CUDA_VISIBLE_DEVICES=$1
shift 1
data_version=$1
dataset=$2
raw_data=$3
A=$4
B=$5
METHOD=$6
filterLabel=$7
shift 7
extra_params="${@}"
convert () {
    data_tools="./Tools/data"
	perl $data_tools/convert_format.pl $1 $2 $3
	perl $data_tools/convert_format.pl $4 $5 $6
}

#work_dir="/mnt/t-sapuro/XC"
work_dir="/workspace/"
data_dir="${work_dir}/data/${dataset}"
model_data_dir="${work_dir}/data/${data_version}/${dataset}"
mkdir -p "${model_data_dir}"
train_file="${data_dir}/train.txt"
test_file="${data_dir}/test.txt"
FILE='score.txt'
    
trn_ft_file="${model_data_dir}/trn_X_Xf.txt"
trn_lbl_file="${model_data_dir}/trn_X_Y.txt"
tst_ft_file="${model_data_dir}/tst_X_Xf.txt"
tst_lbl_file="${model_data_dir}/tst_X_Y.txt"
# convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}

ln -s ${data_dir}/* "${model_data_dir}"
ls "${work_dir}/data/"
if [ $METHOD == 'XMLCNN' ]
then
    echo "Runn:ing XMLCNN"
    cd "./XML-CNN"
    ./run_${data_version}.sh $dataset $raw_data $work_dir $data_version
    cd -
fi


if [ $METHOD == 'AttentionXML' ]
then
    echo "Running AttentionXML"
    cd "./AttentionXML"
    ./run_${data_version}.sh $dataset $raw_data $work_dir $data_version
    cd -
fi

if [ $METHOD == 'MACH' ]
then
    echo "Running MACH"
    #echo "Not Running MACH"
    cd "./MACH"
    ./run.sh $dataset $work_dir ${data_version} ${extra_params}
    cd -
    FILE='score.npz'
fi

fetch_meta_XMLCNN(){
    echo -ne ","
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$5}')
}


fetch_meta_AttentionXML(){
    echo -ne ","
    echo -ne $(head -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$4}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f",$5}')
}


fetch_meta_MACH(){
    pred_time=$(cat ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | grep "pred time (MSec)" | awk -F ' ' '{print $NF}')
    train_time=$(cat ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | grep "train time" | awk -F ' ' '{print $NF}')
    model_size=$(cat ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | grep "Model Size (MB)" | awk -F ' ' '{print $NF}')
    echo "${model_size},${train_time},${pred_time}" | awk -F ',' '{printf "%0.2f,%0.2f,%0.2f",$1,$2,$3}'
}

evaluate(){
    args="${filterLabel} \
          ${model_data_dir} \
          ${model_data_dir}/tst_X_Y.txt \
          ${model_data_dir}/trn_X_Y.txt \
          $A $B 1"
    python3 -u -W ignore Tools/python/evaluate.py ${args} ${score_file}
}

fetch_p_data_rest(){
    train_file=$trn_lbl_file
    test_file=$tst_lbl_file
    score_file="${score_dir}/${METHOD}/${FILE}"
    A=$A
    B=$B
    echo -ne $(evaluate)
    echo -ne $(fetch_meta_${METHOD})
}


if [ ! -e "${work_dir}/results/${data_version}/${dataset}/summary_full.txt" ]; then
    echo "Algo P1 P3 P5 N1 N3 N5 PSP1 PSP3 PSP5 PSN1 PSN3 PSN5 MODELSIZE TRNTIME PREDTIME" |tee "${work_dir}/results/${data_version}/${dataset}/summary_full.txt"
fi

trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"

#train_file="${work_dir}/data/${dataset}/train.txt"
#test_file="${work_dir}/data/${dataset}/test.txt"
score_dir="${work_dir}/results/${data_version}/${dataset}"

echo $METHOD","$(fetch_p_data_rest) | tee -a "${work_dir}/results/${data_version}/${dataset}/summary_full.txt"
#score_file="${score_dir}/${METHOD}/${FILE}"
#echo "$score_file"
#rm -rf $model_data_dir
#cd -
