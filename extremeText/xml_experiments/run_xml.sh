#!/usr/bin/env bash
dset=$1
data_version=$2
work_dir=$3
file_names=($(echo $dset |tr "/" "\n"))
source run_${file_names[-1]}_ensemble.sh
# source run_${file_names[-1]}.sh
SED=sed
if [ $(uname -s) == Darwin ]; then
    SED=gsed
fi

function xml_dataset2ft {
    FILE="$1"
    # Extract metadata
    # remove_invalid $FILE
    INFO=$(head -n 1 $FILE | grep -o "[0-9]\+")
    INFOARRAY=($INFO)
    echo ${INFOARRAY[0]} >> ${FILE}.examples
    echo ${INFOARRAY[1]} >> ${FILE}.features
    echo ${INFOARRAY[2]} >> ${FILE}.labels
    echo "${INFOARRAY[0]} EXAMPLES, ${INFOARRAY[1]} FEATURES, ${INFOARRAY[2]} LABELS"

    # Delete first line
    $SED -i "1d" $FILE
    bash datasets4fastText/tools/libsvm2ft.sh $FILE $2
    wait
}


THREADS=$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')

SCRIPT_DIR=$( dirname "${BASH_SOURCE[0]}" )
BIN=${SCRIPT_DIR}/../extremetext

if [ ! -e ${BIN} ]; then
    cd ${SCRIPT_DIR}/..
    make -j
    cd -
fi

TRAIN="${work_dir}/data/${data_version}/${dset}/train.txt"
TEST="${work_dir}/data/${data_version}/${dset}/test.txt"

xml_dataset2ft $TRAIN "train"
xml_dataset2ft $TEST "test"

model_dir="${work_dir}/models/${data_version}/$dset/XT"
result_dir="${work_dir}/results/${data_version}/$dset/XT"
# # Model training
mkdir -p $model_dir $result_dir
MODEL="${model_dir}/${FILES_PREFIX}_$(echo $PARAMS | tr ' ' '_')"

# if [ ! -e ${MODEL}.bin ]; then
# STARTTIME=$(date +%s)
# time $BIN supervised -input $TRAIN -output $MODEL -loss plt $PARAMS -thread $THREADS
# ENDTIME=$(date +%s)
# TIMELAPSED=$(($ENDTIME - $STARTTIME))
# Test model
# time $BIN test ${MODEL}.bin ${TEST} 1
# time $BIN test ${MODEL}.bin ${TEST} 3
# time $BIN test ${MODEL}.bin ${TEST} 5

# echo "Model: ${MODEL}.bin" > $result_dir'/log_train.txt'
# echo "Model size: $(ls -lh ${MODEL}.bin | grep -E '[0-9\.,]+[BMG]' -o)" >> $result_dir'/log_train.txt'
# echo "Training time is: $TIMELAPSED" >> $result_dir'/log_train.txt'
# exit

# Model quantization
#if [ ! -e ${MODEL}.ftz ]; then
#    time $BIN quantize -output $MODEL -input $TRAIN -thread $THREADS $QUANTIZE_PARAMS
#fi

#time $BIN test ${MODEL}.ftz ${TEST} 1
#time $BIN test ${MODEL}.ftz ${TEST} 3
#time $BIN test ${MODEL}.ftz ${TEST} 5

#echo "Quantized model: ${MODEL}.ftz"
#echo "Quantized model size: $(ls -lh ${MODEL}.ftz | grep -E '[0-9\.,]+[BMG]' -o)"

# Get probabilities for labels in the file
# time $BIN get-prob ${MODEL}.bin ${TEST} ${MODEL}_test.prob ${THREADS}

# Predict labels and get probabilities for labels in the file
instances=$(head -1 $TEST.examples)
labels=$(head -1 $TEST.labels)
STARTTIME=$(date +%s)
time $BIN predict-prob ${MODEL}.bin ${TEST} 10 0 "$result_dir/output.txt" 1 | sed -e 's/__label__\([0-9]*\) \([.0-9]*\)/\1:\2/g' >"$result_dir/score.txt"
ENDTIME=$(date +%s)
TIMELAPSED=$(awk -v var1=$STARTTIME -v var2=$ENDTIME -v var3=$instances 'BEGIN { print ( ( var2 - var1 ) * 1000 / var3 ) }')
echo "Prediction time is: $TIMELAPSED" > $result_dir'/log_predict.txt'
sed -i "1s/^/${instances} ${labels}\n/" "$result_dir/score.txt"
