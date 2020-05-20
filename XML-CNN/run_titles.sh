#!/bin/bash

: '
Use 100 for AmazonTitles-670K
Use 10 for WikiTitles-500K
Use 10 for Wikiseealso-350K
# '

dataset=$1
rawdata=$2
work_dir="${3}"
dataversion="${4}"
DATABUILD='xml_cnn.py'
CONVERT='../tools/data/convert_format.pl'
FETCHDATA='../tools/data/fetch_data.py'
w2v_dir=$work_dir/word2vec_models
sequence_length=10

root_data="$work_dir/data/${dataversion}"
data_dir="${root_data}/${dataset}"
raw_data_dir="$work_dir/RawData/${rawdata}"
result_dir="$work_dir/results/${dataversion}/${dataset}/XMLCNN"
model_dir="$work_dir/models/${dataversion}/${dataset}/XMLCNN"
mkdir -p $result_dir $model_dir $root_data

echo $data_dir

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
	get_data_text "train"
	get_data_text "test"
	python $DATABUILD $1
	rm -rf ${data_dir}/train_raw_full.txt
	rm -rf ${data_dir}/test_raw_full.txt
}


prep_data $data_dir

source $HOME/scratch/anaconda3/etc/profile.d/conda.sh
conda activate xml-cnn
module load compiler/cuda/7.5/compilervars
module load lib/cudnn_cu-9.2/7.1.4/precompiled 

LABELS=$( head -n 1 $root_data/$dataset/test.txt | awk -F ' ' '{print $3}')
FEATURES=$( head -n 1 $root_data/$dataset/test.txt | awk -F ' ' '{print $2}')
echo $LABELS, $FEATURES

THEANO_FLAGS=floatX=float32,device=cuda0 python train.py \
										--mode train\
										--data_path $root_data/$dataset/xml_cnn.p \
										--sequence_length $sequence_length \
										--embedding_dim 300 \
										--num_filters 32 \
										--pooling_units 32 \
										--pooling_type max \
										--hidden_dims 512 \
										--model_variation pretrain \
										--pretrain_type glove \
										--batch_size 128 \
										--num_epochs 35 \
										--vocab_size $FEATURES \
										--labels $LABELS \
										--model_dir $model_dir \
										--results_dir $result_dir \
										--w2v_model_dir $w2v_dir

THEANO_FLAGS=floatX=float32,device=cuda0 python train.py \
										--mode predict\
										--data_path $root_data/$dataset/xml_cnn.p \
										--sequence_length $sequence_length \
										--embedding_dim 300 \
										--num_filters 32 \
										--pooling_units 32 \
										--pooling_type max \
										--hidden_dims 512 \
										--model_variation pretrain \
										--pretrain_type glove \
										--batch_size 128 \
										--num_epochs 35 \
										--epoch_idx 35 \
										--vocab_size $FEATURES \
										--labels $LABELS \
										--model_dir $model_dir \
										--results_dir $result_dir \
										--w2v_model_dir $w2v_dir

conda deactivate
