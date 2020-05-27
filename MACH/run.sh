##### Enabling the import of a function used in evaluation #####
cd ./util
make
cd ../
dataset=$1
workspace="$2"
data_version="$3"
command -v module >/dev/null 2>&1 && module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
command -v module >/dev/null 2>&1 && module load pythonpackages/3.6.0/pandas/0.23.4/gnu
command -v module >/dev/null 2>&1 && module load pythonpackages/3.6.0/numpy/1.16.1/gnu

# module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
# module load pythonpackages/3.6.0/pandas/0.23.4/gnu
# module load pythonpackages/3.6.0/numpy/1.16.1/gnu

export PYTHONPATH="${PYTHONPATH}:util"
##### Move into the respective folder #####
B=$4
echo "Number of B", $B
R=32
num_gpus=4
models_per_gpu=1
# python
# exit
#echo "Hi"
RbyNumGPU=$( echo "$R / ( $num_gpus )" | bc )
Rbatches=$( echo "$R / ( $num_gpus * ${models_per_gpu} )" | bc )
gpufrac=$(awk "BEGIN {print (0.9)/ ${models_per_gpu}}")

data_dir="${workspace}/data/$dataset"
model_dir="${workspace}/models/${data_version}/$dataset/MACH"
result_dir="${workspace}/results/${data_version}/$dataset/MACH"
temp_dir="$workspace/data/mach_temp/${data_version}/$dataset"
mkdir -p $temp_dir
mkdir -p $result_dir
mkdir -p $model_dir"/b_$B"
ln -s $data_dir/* $temp_dir
#ls -R /
#locate cuda | grep /cuda$
echo "Here2"
export PATH=/usr/local/cuda-10.0/bin:$PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# python --version
# echo "Here3"
# echo "$PATH"
# echo "Here4"
# find / -name nvcc 2>/dev/null
# echo "Here5"
# find / -type d -name cuda 2>/dev/null
# nvcc --version
# cat /usr/local/cuda/version.txt
echo "here"
pwd
#ls -R "$workspace/data/"
num_classes=$(head -n 1 $temp_dir/train.txt| awk -F ' ' '{print $3}')
num_features=$(head -n 1 $temp_dir/train.txt| awk -F ' ' '{print $2}')
num_train=$(head -n 1 $temp_dir/train.txt| awk -F ' ' '{print $1}')
num_test=$(head -n 1 $temp_dir/test.txt| awk -F ' ' '{print $1}')

#####  Build lookups for classes #####
mkdir -p "$temp_dir/b_$B/lookups"
# exit

##### Training multiple repetitions simulataneously #####
#script="--batch_size=100 --n_epochs=60 --load_epoch=0 \
#        --data_dir $temp_dir --result_dir ${result_dir} --model_dir $model_dir --B $B \
#        --n_train $num_train --num_features ${num_features}"

script="--batch_size=30 --n_epochs=60 --load_epoch=0 \
        --data_dir $temp_dir --result_dir ${result_dir} --model_dir $model_dir --B $B \
        --n_train $num_train --num_features ${num_features}"

run_learner(){
    args=$1
    echo $args
    python -u train_single.py ${args} > "${result_dir}/log_train_rep_${learner}.txt"
}

run_job(){
    gpu_idx=$1
    local_learner=0
    RBatch=0
    for((RBatch=0; RBatch<$Rbatches; RBatch++));
    do
        for((lerners_pergpu=0; lerners_pergpu<$models_per_gpu; lerners_pergpu++));
        do
            learner=$( echo "(${gpu_idx} * ${RbyNumGPU} + ${RBatch} * ${models_per_gpu} + ${lerners_pergpu})" | bc)
            args="${script} --repetition=${learner} --B=${B} \
                    --gpu=${gpu_idx} --gpu_usage=${gpufrac}"
            run_learner "$args" &
        done
        wait
    done
}

echo "Running representations $R" > "${result_dir}/log_train_args.txt"
pids=""
train(){
    mkdir -p "$result_dir/b_${B}/lookups"
    python build_index.py --write_loc "$result_dir/b_${B}/lookups" --num_classes $num_classes --B $B --R $R
    for((gpu_idx=0; gpu_idx<$num_gpus; gpu_idx++));
    do
        run_job $gpu_idx &
    done 
    wait
}
predict(){
    #models_per_gpu=32
    args="--R $R --data_dir $temp_dir --model_dir $model_dir \
        --num_features ${num_features} --num_samples ${num_test} \
        --result_dir ${result_dir} --num_classes $num_classes \
        --B $B --num_gpus ${num_gpus} --gpu ${CUDA_VISIBLE_DEVICES}\
        --models_per_gpu ${RbyNumGPU} --batch_size 32"
    echo $args
    python -u eval.py $args 2>&1|tee "$result_dir/log_predict.txt"
    # python eval_old.py $args |tee "$result_dir/log_predict.txt"
}

train

predict

##### Get precision@1,3,5 #####

rm -rf $temp_dir
