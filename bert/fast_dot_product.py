import sys
import os
import subprocess
import pdb
import numpy as np
import json
from scipy import sparse
from xclib.data import data_utils as du
from sklearn.preprocessing import normalize

def get_emb(file):
	mat=[]
	with open(file) as f:
		line = f.readline().strip()
		rows = int(line.split(" ")[0])
		cols = int(line.split(" ")[1])
		for ind in range(rows):
			line = f.readline().strip()
			arr = line.split(" ")
			arr = [float(el) for el in arr]
			mat.append(arr)
		line = f.readline().strip()
		if line:
			pdb.set_trace()
	mat = np.matrix(mat)
	return mat

if __name__ == "__main__":
	dset="new-eurlex"
	path ="/mnt/e/Analysis/Data/"+dset+"/"
	emb_file = path+"tst_X_Xf_fast.txt"
	dp_emb = get_emb(emb_file)	
	dp_emb = np.matrix(normalize(dp_emb, norm='l2', axis=1))
	
	emb_file = path+"Y_token_uni_fast.txt"
	lab_emb = get_emb(emb_file)
	lab_emb = np.matrix(normalize(lab_emb, norm='l2', axis=1))

	score = dp_emb * (lab_emb.transpose())
	score=sparse.csr_matrix(score)
	du.write_sparse_file(score,"/mnt/e/Analysis/Results/"+dset+"/knn_fast.txt")	
	