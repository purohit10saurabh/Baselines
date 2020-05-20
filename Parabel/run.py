from __future__ import print_function

import sys
import os

from p_mat import *
from utils import *
from compute_all_metrics import *
import subprocess
import pdb

os.environ["EXP_DIR"] = "/mnt/c/Users/t-nilgup/Desktop"

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

	print("Running command : " + command)

	for arg in args:
		display_command += " " + arg
	bash_command += display_command

	if(os.system(bash_command) == 0):
		print(bcolors.OKGREEN + "Successfully" + bcolors.ENDC + " ran bash command : " + display_command)
	else:
		print(bcolors.FAIL + "Failure" + bcolors.ENDC + " while running bash command : " + display_command)
		exit(0)

dataset = "Weighted-EURLex-4K"
if "-d" in sys.argv:
	d_id = sys.argv.index("-d") + 1
	dataset = sys.argv[d_id]

options = "btpe" # b - build, t - train, p - predict, e - evaluate scores
if "-o" in sys.argv:
	op_id = sys.argv.index("-o") + 1
	options = sys.argv[op_id]

results_dir = "./Results/%s" %( dataset )
model_dir = "./Results/%s/model" %( dataset )
if not os.path.exists( model_dir ):
    os.makedirs( model_dir )
sys.stdout = Logger( os.path.join( results_dir, "log.txt" ) )

weighted = int("Weighted" in dataset)
if "-w" in sys.argv:
	w_id = sys.argv.index("-w") + 1
	weighted = int(sys.argv[w_id])

per_label_predict = int("--per_label_predict" in sys.argv)

data_dir = os.path.join( os.getenv( "EXP_DIR" ), "Experiments", "Datasets", dataset )
trn_X_Xf_file = os.path.join( data_dir, "trn_X_Xf.txt" )
tst_X_Xf_file = os.path.join( data_dir, "tst_X_Xf.txt" )
trn_X_Y_file = os.path.join( data_dir, "trn_X_Y.txt" )
tst_X_Y_file = os.path.join( data_dir, "tst_X_Y.txt" )

score_file = "%s/score_mat.txt" %(results_dir)

if "b" in options:
	print(bcolors.HEADER + "\n-------------------- BUILDING... --------------------" + bcolors.ENDC); sys.stdout.flush()
	runBashCommand("make", [])
	print(bcolors.HEADER + "-----------------------------------------------------\n" + bcolors.ENDC); sys.stdout.flush()

if "t" in options:
	print(bcolors.HEADER + "\n-------------------- TRAINING... --------------------" + bcolors.ENDC); sys.stdout.flush()
	train_args = [	model_dir, str(0), "--trn_ft_file", trn_X_Xf_file, "--trn_lbl_file", trn_X_Y_file, 
					"-w", str(weighted), 
					"-t", str(3), 
					"-m", str(100), 
					"-k", str(weighted), 
					"-tcl", str(0.05), 
					"-ecl", str(0.1),
					"-r", str(1)	]
	train_command = "./parabel_train"
	runBashCommand(train_command, train_args)
	print(bcolors.HEADER + "-----------------------------------------------------\n" + bcolors.ENDC); sys.stdout.flush()

if "p" in options:
	print(bcolors.HEADER + "\n-------------------- PREDICTING... --------------------" + bcolors.ENDC); sys.stdout.flush()
	predict_args = [model_dir, str(0), score_file, "--tst_ft_file", tst_X_Xf_file, "-t", str(3), "-p", str(per_label_predict)]
	predict_command = "./parabel_predict"
	runBashCommand(predict_command, predict_args)
	print(bcolors.HEADER + "-------------------------------------------------------\n" + bcolors.ENDC); sys.stdout.flush()

if "e" in options:
	print(bcolors.HEADER + "\n-------------------- EVALUATING... --------------------" + bcolors.ENDC); sys.stdout.flush()
	K = 5
	score_mat = read_text_mat(score_file)
	tst_X_Y = read_text_mat(tst_X_Y_file)

	if weighted:
		p_compute_all_metrics( score_mat, tst_X_Y, K )
	else:
		runBashCommand("python evaluate.py", [tst_X_Y_file, score_file, str(K)])

	print(bcolors.HEADER + "-------------------------------------------------------\n" + bcolors.ENDC); sys.stdout.flush()
