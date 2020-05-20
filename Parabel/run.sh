arguments=${@:2}
dataset=$1
logfile=./Results/$dataset/log.txt
stdbuf -o0 python2.7 run.py -d $dataset $arguments |& tee -a $logfile