#!/bin/bash
# module load compiler/gcc/5.1.0/compilervars
dataset=$1
data_version=$2
A=$3
B=$4
METHOD=$5

emb=300
prefix=$(echo $dataset | tr '-' "d")

convert () {
    data_tools="./Tools/data"
	perl $data_tools/convert_format.pl $1 $2 $3
	perl $data_tools/convert_format.pl $4 $5 $6
}

work_dir="${HOME}/scratch/XC"
data_dir="${work_dir}/data/${dataset}"
model_data_dir="${work_dir}/data/${data_version}/${dataset}"
mkdir -p "${model_data_dir}"
embedding="fasttextB_embeddings_${emb}d"
train_file="${data_dir}/train.txt"
test_file="${data_dir}/test.txt"
    
trn_ft_file="${model_data_dir}/trn_X_Xf.txt"
trn_lbl_file="${model_data_dir}/trn_X_Y.txt"
tst_ft_file="${model_data_dir}/tst_X_Xf.txt"
tst_lbl_file="${model_data_dir}/tst_X_Y.txt"

if [ ! -e "${trn_ft_file}" ] 
then
    echo "Converting dataset"
    convert ${train_file} ${trn_ft_file} ${trn_lbl_file} ${test_file} ${tst_ft_file} ${tst_lbl_file}
fi

ln -s ${train_file} "${model_data_dir}/train.txt"
ln -s ${test_file} "${model_data_dir}/test.txt"
cp "${data_dir}/filter_labels_test.txt" "${model_data_dir}/filter_labels_test.txt"

if [ $data_version == 'dense' ]
then
    echo "Create it Wohooo!"
    sh gen_dense_features.sh "${model_data_dir}" "${data_dir}/${embedding}.npy" "wt_avg" "trn"
    sh gen_dense_features.sh "${model_data_dir}" "${data_dir}/${embedding}.npy" "wt_avg" "tst"
fi
wait


num_test_samples=$(head -n 1 $tst_ft_file |awk -F ' ' '{print $1}')
cwd=$(pwd)

if [ $METHOD == 'Parabel' ]
then
    echo "Running Parabel"
    num_trees=3
    ./run_parabel.sh $work_dir $dataset $data_version $num_trees
fi

if [ $METHOD == 'Bonsai' ]
then
    echo "Running Bonsai"
    num_trees=3
    ./run_bonsai.sh $work_dir $dataset $data_version $num_trees
fi

if [ $METHOD == 'FastXML' ]
then
    echo "Running FastXML"
    ./run_fastxml.sh $work_dir $dataset $data_version 
fi

if [ $METHOD == 'PLT' ]
then
    echo "Running PLT"
    ./run_plt.sh $work_dir $dataset $data_version 
fi

if [ $METHOD == 'ParabelOld' ]
then
    echo "Running Parabel_old"
    num_trees=3
    ./run_parabel_old.sh $work_dir $dataset $data_version $num_trees
fi


if [ $METHOD == 'PfastreXML' ]
then
    echo "Running PfastreXML"
    num_trees=50
    ./run_pfastrexml.sh $work_dir $dataset $data_version $num_trees $A $B
fi

if [ $METHOD == 'AnneXML' ]
then
    echo "Running AnneXML"
    ./run_annexml.sh $work_dir $dataset $data_version
fi

if [ $METHOD == 'Slice' ]
then
    echo "Running Slice"
    ./run_slice.sh $dataset $data_version $work_dir
    rm -rf "${work_dir}/models/dense/$dataset/Slice/tmp"
fi


if [ $METHOD == 'ova' ]
then
    echo "Running ova"
    ./run_ova.sh $dataset $data_version $work_dir "ova"
fi


if [ $METHOD == 'XT' ]
then
    echo "Running XT"
    cd ./extremeText/xml_experiments
    ./run_xml.sh $dataset $data_version $work_dir
    cd -
fi

fetch_meta_Slice(){
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | sed '5q;d' | awk -F ' ' '{printf "%0.2f,",$4}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$4}')
    echo $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$5}')
}

fetch_meta_Parabel(){
    echo -ne $(tail -2 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$5}')
    echo $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$7}')
}


fetch_meta_ParabelOld(){
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '1q;d' | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '2q;d' | awk -F ' ' '{printf "%0.2f",$3}')
}


fetch_meta_Bonsai(){
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt)","
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(head -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt)
}

fetch_meta_FastXML(){
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%.2f",$3}')","
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '2q;d' | awk -F ' ' '{printf "%0.2f",$3}')
}

fetch_meta_PLT(){
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt )","
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt )
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '2q;d' | awk -F ' ' '{printf "%0.2f",$1}')
}

fetch_meta_AnneXML(){
    echo -ne $(sed -n 2p ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f,",$2}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '2q;d' | awk -F ' ' '{printf "%0.2f",$4}')
}



fetch_meta_XT(){
    echo -ne $(sed -n 2p ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{print $3}' |tr 'MG' ',')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$4}')
    echo -ne $(head -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$4}')
}


fetch_meta_ova(){
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,%0.2f,",$8,$4}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | awk -F ' ' '{printf "%0.2f",$4}')
}


fetch_meta_PfastreXML(){
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '1q;d' | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tail -1 ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_train.txt | awk -F ' ' '{printf "%0.2f,",$3}')
    echo -ne $(tac ${work_dir}/results/${data_version}/${dataset}/$METHOD/log_predict.txt | sed '2q;d' | awk -F ' ' '{printf "%0.2f",$3}')
}

evaluate(){
    args="filter_labels_test.txt \
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

train_file="${work_dir}/data/${data_version}/${dataset}/train.txt"
test_file="${work_dir}/data/${data_version}/${dataset}/test.txt"
score_dir="${work_dir}/results/${data_version}/${dataset}"
FILE='score.txt'

echo $METHOD","$(fetch_p_data_rest) | tee -a "${work_dir}/results/${data_version}/${dataset}/summary_full.txt"
rm -rf $model_data_dir
