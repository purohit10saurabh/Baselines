#!/bin/bash

CONVERT='../tools/data/convert_format.pl'
FETCHDATA='../tools/data/fetch_data.py'

dataset=$1
rawdata=$2
workdir=$3
dataversion=$4

root_data="$workdir/data/${dataversion}"
data_dir="${root_data}/${dataset}"
raw_data_dir="$workdir/RawData/${rawdata}"
result_dir="$workdir/results/${dataversion}/${dataset}/AttentionXML"
model_dir="$workdir/models/${dataversion}/${dataset}/AttentionXML"
mkdir -p $result_dir $model_dir $root_data

rm -rf results data models

ln -s "${result_dir}" "results"
ln -s "${root_data}" "data"
ln -s "${model_dir}" "models"
NUM_LABELS=""
prep_data(){
	get_data_text(){
		prefix=$1
		awk -F: '{ st = index($0,"->");print substr($0,st+2)}' "${raw_data_dir}/${prefix}_map.txt" | tr '_' ' ' > "${data_dir}/${prefix}_txt"
		perl $CONVERT "${data_dir}/${prefix}.txt" "$data_dir/${prefix}_X_X.txt" "$data_dir/${prefix}_X_Y.txt"
		args="${data_dir}/${prefix}_raw_ids.txt \
				${data_dir}/${prefix}_txt \
				${data_dir}/${prefix}_X_Y.txt \
				${data_dir} \
				${prefix}"
		python $FETCHDATA $args
		rm -rf "${data_dir}/${prefix}_txt" "$data_dir/${prefix}_X_X.txt"
	}
    awk -F: '{ st = index($0,"->");print substr($0,st+2)}' "${raw_data_dir}/train_map.txt" | tr '_' ' ' > "${data_dir}/raw_full.txt"
	get_data_text "train"
	get_data_text "test"
    NUM_LABELS=$(head -n 1 ${data_dir}/train.txt | awk -F ' ' '{print $3}')
    sed '1d' "${data_dir}/train.txt" > "${data_dir}/train_new.txt" 
    mv "${data_dir}/train_new.txt" "${data_dir}/train.txt"
}

tokeinze(){
    prep_data
    args=" --text-path ${data_dir}/raw_full.txt \
            --tokenized-path ${data_dir}/full_texts.txt \
            --vocab-path ${data_dir}/vocab.npy \
            --emb-path ${data_dir}/emb_init.npy \
            --w2v-model ${workdir}/models/w2v/glove.840B.300d.gensim"

    PYTHONPATH=src python src/deepxml/preprocess.py $args
                
    args="--text-path ${data_dir}/test_raw_full.txt \
            --tokenized-path ${data_dir}/test_texts.txt \
            --label-path ${data_dir}/test_labels.txt \
            --vocab-path ${data_dir}/vocab.npy"
    PYTHONPATH=src python src/deepxml/preprocess.py $args
                    
    args="--text-path ${data_dir}/train_raw_full.txt \
            --tokenized-path ${data_dir}/train_texts.txt \
            --label-path ${data_dir}/train_labels.txt \
            --vocab-path ${data_dir}/vocab.npy"
    PYTHONPATH=src python src/deepxml/preprocess.py $args

    rm -rf ${data_dir}/test_raw_full.txt
    rm -rf ${data_dir}/train_raw_full.txt
}


train_predict_single(){
    args="--data-cnf configure/datasets/${dataset}.yaml \
         --model-cnf configure/models/FastAttentionXML-${dataset}.yaml"
    PYTHONPATH=src python src/deepxml/tree.py $args
}

merge_score_mats(){
    python merge_score.py $result_dir'/scorenew.txt' $result_dir'/score.txt-Tree-0' $result_dir'/score.txt-Tree-1' $result_dir'/score.txt-Tree-2'
    mv $result_dir'/scorenew.txt' $result_dir'/score.txt'
}

train_predict_ensemble(){
    args="--data-cnf configure/datasets/${dataset}.yaml \
         --model-cnf configure/models/FastAttentionXML-${dataset}.yaml --tree-id 0 --num_labels ${NUM_LABELS}"
    PYTHONPATH=src python src/deepxml/tree.py $args
    args="--data-cnf configure/datasets/${dataset}.yaml \
         --model-cnf configure/models/FastAttentionXML-${dataset}.yaml --tree-id 1 --num_labels ${NUM_LABELS}"
    PYTHONPATH=src python src/deepxml/tree.py $args
    args="--data-cnf configure/datasets/${dataset}.yaml \
         --model-cnf configure/models/FastAttentionXML-${dataset}.yaml --tree-id 2 --num_labels ${NUM_LABELS}"
    PYTHONPATH=src python src/deepxml/tree.py $args
    merge_score_mats
}

tokeinze
train_predict_ensemble | tee -a $result_dir'/log_train.txt'
# train_predict_ensemble