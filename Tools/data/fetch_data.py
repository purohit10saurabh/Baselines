from xclib.data import data_utils as du
import sys
import os
import numpy as np

valid_ids = np.loadtxt(sys.argv[1], dtype=int)
raw_titles = sys.argv[2]
labels = du.read_sparse_file(sys.argv[3], force_header=True)
output_dir = sys.argv[4]
prefix= sys.argv[5]

with open(os.path.join(output_dir, "{}_raw_full.txt".format(prefix)), 'w') as ft:
    title = open(raw_titles, 'r', encoding='latin1').readlines()
    if len(sys.argv)>6:
        raw_text = sys.argv[6]
        print("Using full text")
        text = open(raw_text, 'r', encoding='latin1').readlines()
        for valid in valid_ids:
            ft.write("%s /SEP/ %s %s\n"%(title[valid].strip(), title[valid].strip(), text[valid].strip()))
    else:
        for valid in valid_ids:
            ft.write(title[valid])


with open(os.path.join(output_dir, "{}_labels.txt".format(prefix)), 'w') as ft:
    for valid in range(labels.shape[0]):
        ft.write("%s\n"%(" ".join(list(map(lambda x: str(x), labels[valid].indices.tolist())))))