import sys
import os
import subprocess
import pdb
import numpy as np
import json
from scipy import sparse
from xclib.data import data_utils as du
from sklearn.preprocessing import normalize

def read_emb(line):
	dp_emb=json.loads(line)
	index = dp_emb['linex_index']		
	dp_emb = dp_emb['features']
	if len(dp_emb)!=1:
		pdb.set_trace()
	dp_emb=dp_emb[0]
	if dp_emb['token']!='[CLS]':
		pdb.set_trace()
	dp_emb=dp_emb['layers']
	if len(dp_emb)!=1:
		pdb.set_trace()
	dp_emb=dp_emb[0]
	if dp_emb['index']!=-1:
		pdb.set_trace()
	dp_emb=dp_emb['values']
	return [index,dp_emb]

def get_emb(dp_emb_file):
	embs=[]
	ind=0
	with open(dp_emb_file) as f:
		while True:
			line=f.readline().strip()
			if not line:
				break
			[index,dp_emb]=read_emb(line)			
			if ind!=index:
				pdb.set_trace()
			embs.append(dp_emb)
			ind+=1
	emb=np.matrix(embs)
	return emb

if __name__ == "__main__":
	dset="new-eurlex"
	path ="/mnt/t-sapuro/XC/data/"+dset+"/"
	emb_file = path+"tst_X.content_hr_bert.txt"
	dp_emb = get_emb(emb_file)	
	#dp_emb = np.matrix(normalize(dp_emb, norm='l2', axis=1))
	
	emb_file = path+"Y.title_bert.txt"
	#emb_file = path+"my_Y_bert.txt"
	lab_emb = get_emb(emb_file)	
	#lab_emb = np.matrix(normalize(lab_emb, norm='l2', axis=1))

	score = dp_emb * (lab_emb.transpose())
	score=sparse.csr_matrix(score)
	du.write_sparse_file(score,path+'knn_bert.txt')
	#pdb.set_trace()
