import sys
import os
import subprocess
import pdb

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def runBashCommand(command, args):
	bash_command = bash_command = "/usr/bin/time -f '\n" + bcolors.OKBLUE + "Total Time : %E (mm:ss)\nMax RAM : %M kB" + bcolors.ENDC + "\n' "
	display_command = command

	for arg in args:
		display_command += " " + arg
	bash_command += display_command

	print("Running command : " + display_command)
	
	if(os.system(bash_command) == 0):
		print(bcolors.OKGREEN + "Successfully" + bcolors.ENDC + " ran bash command : " + display_command)
	else:
		print(bcolors.FAIL + "Failure" + bcolors.ENDC + " while running bash command : " + display_command)
		exit(0)

#workspace="/workspace/"
workspace="/workspace/"
path=workspace+"data/bert/"
BERT_BASE_DIR=path+"uncased_L-12_H-768_A-12/"

input_file= workspace+"data" + sys.argv[1] + '/' + sys.argv[2] + ".txt"
out_dir = workspace+"results/" + sys.argv[1]
comm = "mkdir "+out_dir
if(os.system(comm) == 0):
	print("success in mkdir")
else:
	print("error in mkdir")
output_file= out_dir + '/' + sys.argv[2] + "_bert.txt"

vocab_file=BERT_BASE_DIR+"vocab.txt"
bert_config_file=BERT_BASE_DIR+"bert_config.json"
init_checkpoint=BERT_BASE_DIR+"bert_model.ckpt"

if __name__ == "__main__":
	bert_args = [ 	"--input_file", input_file,
					"--output_file", output_file,
					"--vocab_file", vocab_file,
					"--bert_config_file", bert_config_file,	
					"--init_checkpoint", init_checkpoint,
					"--layers", '-1',	
					"--max_seq_length", str(256),
					"--batch_size", str(8),
				]

	for i in range(len(bert_args)):
		arg = bert_args[i]
		#if ((arg[0] == '-') and (arg in sys.argv)):
		#	bert_args[i+1] = str(sys.argv[sys.argv.index(arg) + 1])

	bert_command = "python ./bert/extract_features.py"
	runBashCommand(bert_command, bert_args)
	