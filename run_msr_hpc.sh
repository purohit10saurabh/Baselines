#!/bin/bash

#### This type of setup will expect `run_msr_hpc.sh` to be present in the root folder of the to be run GitHub repo.
#### Below is an example:

#### Requirement as environment variables: 
# AZURE_BLOB_ROOT: storage root for taking data from (will already be set from docker instance)
# AZURE_SECRET_KEY: azure key for communicating (will already be set from docker instance)
# MODEL: model type
# DATASET: dataset to run model on
# RUN: run number

root_blob=$AZURE_BLOB_ROOT
secret_key=$AZURE_SECRET_KEY

log_file="log_master"
if [ ! -z "$GITHUB_BRANCH" ]; then log_file="log_${GITHUB_BRANCH}"; fi
log_file="${log_file}_${A}_${method}_${dset}"
# if [ -z "$ENV1" ]; then log_file="${log_file}_env1"; else log_file="${log_file}_no_env1"; fi
# if [ -z "$ENV2" ]; then log_file="${log_file}_env2"; else log_file="${log_file}_no_env2"; fi
# if [ -z "$ENV3" ]; then log_file="${log_file}_env3"; else log_file="${log_file}_no_env3"; fi
log_file="${log_file}.txt"

########## Function which will upload log file to azure storage.
upload_log_file () {
    yes | azcopy --source ./ --destination ${root_blob}/hpc_logs/ --dest-key ${secret_key} --include ${log_file}
}

########## Creating log file
touch ${log_file}

########## printing hardware information
nvidia-smi | tee -a ${log_file}
free -gh | tee -a ${log_file}
lscpu | tee -a ${log_file}

########## Copy datasets from azure storage into docker image
echo "Downloading data from azure blob..." | tee -a ${log_file}
azcopy --destination ../data/ --source ${root_blob}/data/ --source-key ${secret_key} --recursive | tee -a ${log_file}

########## Starting azcopy background process to upload log file every 'x' seconds
while : ; do 
    upload_log_file > /dev/null 2>&1
    sleep 10;
done &
AZCOPY_PID=$!

########## Print current branch
echo "Current branch: " | tee -a ${log_file}
git branch | tee -a ${log_file}

########## Run Code
echo "Running code..." | tee -a ${log_file}
if [ "$GPU" == "" ]; then GPU=0; fi
#python main.py # If this is what you want to do.
./my_run_deeplearning.sh $GPU titles $dset $dset $A $B $method 1 2000 2>&1 | tee -a ${log_file}
#CUDA_VISIBLE_DEVICES=$GPU ./run_all_cpu.sh EURLex-4K sparse ${A} 1.5 Parabel 2>&1 | tee -a ${log_file}

########## Kill azcopy background process and write output to azure file storage one final time
kill -9 $AZCOPY_PID
upload_log_file
yes | azcopy --source ../results/ --destination ${root_blob}/results/ --dest-key ${secret_key} --recursive