import numpy as np
from sklearn.utils import murmurhash3_32 as mmh3
import argparse
from multiprocessing import Pool
import time
import os
parser = argparse.ArgumentParser()
parser.add_argument("--n_cores", default=32, type=int)
parser.add_argument("--B", default=2000, type=int)
parser.add_argument("--R", default=4, type=int)
parser.add_argument("--write_loc", default='data/b_2000/lookups', type=str)
parser.add_argument("--num_classes", default=30938, type=int)
args = parser.parse_args()

num_classes = args.num_classes

B = args.B
R = args.R

def process_r(r):
    bucket_order = np.zeros(num_classes, dtype=int)
    #
    for i in range(num_classes):
        bucket = mmh3(i,seed=r)%B
        bucket_order[i] = bucket
        #
    np.save(os.path.join(args.write_loc, 'bucket_order_'+str(r)+'.npy'), bucket_order)
    del bucket_order
    return None


p = Pool(args.n_cores)

begin_time = time.time()
p.map(process_r, range(args.R))
print('time_elapsed:',time.time()-begin_time)

p.close()
p.join()

