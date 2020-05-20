import sys
import os

num_labels = int(sys.argv[2])
with open(sys.argv[1], "r") as infile:
    with open(os.path.join(os.path.dirname(sys.argv[1]), "score2.txt"), "w") as outfile:
        for line in infile:
            line = filter(lambda x: int(x.split(':')[0]) < num_labels, line.strip().split(" "))
            outfile.write(" ".join(line) + "\n") 
