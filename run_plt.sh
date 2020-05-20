#!/bin/bash

work_dir=$1
dataset=$2
data_version=$3
num_trees=$4

#module load compiler/gcc/4.9.3/compilervars
data_dir="${work_dir}/data/${data_version}/${dataset}"
model_dir="${work_dir}/models/${data_version}/${dataset}/PLT"
result_dir="${work_dir}/results/${data_version}/${dataset}/PLT"
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
trn_ft_lbl_file="${data_dir}/trn_X_XY.txt"
score_file="${result_dir}/score.txt"

DATASET_NAME=$dataset
FILES_PREFIX=$dataset
SED=sed
if [ $dataset == "Eurlex" ]
then
    K=3993
    PARAMS="-l 0.0003 --power_t 0.2 --kary_tree 64 --passes 30 -b 30"
fi

# bash run_xml.sh $DATASET_NAME $FILES_PREFIX $K "$PARAMS"

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
BIN=${SCRIPT_DIR}/extweme_wabbit/vowpalwabbit/vw
NUM_TEST_SAMPLES=$(head -1 $data_dir/tst_X_Xf.txt | awk -F ' ' '{printf "%d",$1}')

# Installing required codebases for PLT
if [ ! -d $SCRIPT_DIR/extweme_wabbit ]
then
    git clone https://github.com/mwydmuch/extweme_wabbit
    cd extweme_wabbit
    make
    make install
    # Since extweme_wabbit places some binaries in the /usr/bin. 
    # Allowing only copying with sudo privileges. Therefore install is run again below.
    echo 'extweme_wabbit places some binaries in the /usr/bin. Please allow Sudo privileges'
    sudo make install 
fi

function xml_dataset4vw {
    FILE="$1"

    # Extract metadata
    INFO=$(head -n 1 $FILE | grep -o "[0-9]\+")
    INFOARRAY=($INFO)
    echo ${INFOARRAY[0]} >> ${FILE}.examples
    echo ${INFOARRAY[1]} >> ${FILE}.features
    echo ${INFOARRAY[2]} >> ${FILE}.labels

    echo "CONVERTING $FILE TO VW FORMAT ..."
    echo "${INFOARRAY[0]} EXAMPLES, ${INFOARRAY[1]} FEATURES, ${INFOARRAY[2]} LABELS"

    # Delete first line
    $SED -i "1d" $FILE

    # Add labels/features separator
    $SED -i "s/\(\(^\|,\| \)[0-9]\{1,\}\)  *\([0-9]\+:\)/\1 | \3/g" $FILE
    $SED -i "s/^ *\([0-9]\+:\)/\| \1/g" $FILE

    # replace 0 with the highest label
    # $SED -i "s/^0[ ,]\(.*\)|/\1 ${INFOARRAY[2]} |/g" $FILE

    # replace comas with spaces
    # $SED -i "s/,/ /g" $FILE
}
# echo "This is the file $data_dir/$FILES_PREFIX/train.txt"
if [ -e "$data_dir/train.txt" ]; then
    # and "./$FILES_PREFIX/${FILES_PREFIX}_test.txt"

    echo "PROCESSING ${FILES_PREFIX} ..."

    cp  "$data_dir/train.txt"  "$data_dir/${FILES_PREFIX}_train"
    xml_dataset4vw "$data_dir/${FILES_PREFIX}_train"

    cp  "$data_dir/test.txt"  "$data_dir/${FILES_PREFIX}_test"
    xml_dataset4vw "$data_dir/${FILES_PREFIX}_test"

    # bash ${SCRIPT_DIR}/../tools/remap_dataset.sh "./$FILES_PREFIX/${FILES_PREFIX}_train" "./$FILES_PREFIX/${FILES_PREFIX}_test"
fi

if [ -e "$data_dir/${DATASET_NAME}_data.txt" ]; then

    echo "PROCESSING ${FILES_PREFIX} ..."

    mv "$data_dir/${DATASET_NAME}_data.txt" "$data_dir/${FILES_PREFIX}_data"
    mv "$data_dir/${FILES_PREFIX}_trSplit.txt" "$data_dir/${FILES_PREFIX}_trSplit"
    mv "$data_dir/${FILES_PREFIX}_tstSplit.txt" "$data_dir/${FILES_PREFIX}_tstSplit"
    xml_dataset4vw "$data_dir/${FILES_PREFIX}_data"

    # bash ${SCRIPT_DIR}/../tools/remap_dataset.sh "./$FILES_PREFIX/${FILES_PREFIX}_data"

    bash ${SCRIPT_DIR}/extweme_wabbit/tools/split_dataset.sh "$data_dir/${FILES_PREFIX}_data" "$data_dir/${FILES_PREFIX}_trSplit" "$data_dir/${FILES_PREFIX}_train"
    bash ${SCRIPT_DIR}/extweme_wabbit/tools/split_dataset.sh "$data_dir/${FILES_PREFIX}_data" "$data_dir/${FILES_PREFIX}_tstSplit" "$data_dir/${FILES_PREFIX}_test"

    # rm "./$FILES_PREFIX/${FILES_PREFIX}_trSplit"
    # rm "./$FILES_PREFIX/${FILES_PREFIX}_tstSplit"
    # rm "./$FILES_PREFIX/${FILES_PREFIX}_data"
fi


TRAIN=$data_dir/${FILES_PREFIX}_train
TEST=$data_dir/${FILES_PREFIX}_test

if [ ! -e $TRAIN ] 
then
    TRAIN=$data_dir/${FILES_PREFIX}_train0
    TEST=$data_dir/${FILES_PREFIX}_test0
fi

MODEL="$model_dir/${FILES_PREFIX}_$(echo $PARAMS | tr ' ' '_')"

start_time=$( date +%s )
time $BIN --plt $K $TRAIN -f $MODEL --sgd $PARAMS --cache_file $data_dir/cache.file | tee $log_train
end_time=$( date +%s )

echo $(( end_time - start_time )) >> $log_train 
start_time=$( date +%s )
time $BIN -t -i $MODEL $TEST --top_k 20 -p $result_dir/score.txt | tee  $log_predict 
end_time=$( date +%s )
total_time=$(( end_time - start_time ))
#cp $result_dir/score.txt $result_dir/score.csv
python ./Tools/python/plt_postprocessing.py $result_dir/score.txt $K
mv $result_dir/score2.txt $result_dir/score.txt

sed -i "1s/^/$NUM_TEST_SAMPLES $K\n/" $result_dir/score.txt
echo "$( printf 'scale=2; %f / %f\n' "$total_time" "$NUM_TEST_SAMPLES" | bc )" >> $log_predict
echo $(ls -l --block-size=M $MODEL | awk -F ' ' '{printf "%d",$4}') >> $log_predict

