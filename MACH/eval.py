# exit(0)
import scipy.sparse as sp
from multiprocessing import Pool
import math
import argparse
import json
import glob
import time
import tensorflow as tf
import sys
import pdb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# pdb.set_trace()
try:
    from util import gather_batch
except:
    print('**********************CANNOT IMPORT GATHER***************************')
    exit()
import numpy as np

# tf.logging.set_verbosity(0)
parser = argparse.ArgumentParser()
parser.add_argument("--R", help="how many repetitions?", default=32, type=int)
parser.add_argument("--B", help="how many buckets?", default=2000, type=int)
parser.add_argument("--gpu", help="which GPU?", default='0,1,2,3', type=str)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--epoch", default='60', type=str)
parser.add_argument("--parallel_across", default='batch',
                    choices=['batch', 'classes'], type=str)
parser.add_argument("--n_threads", default=16, type=int)
parser.add_argument("--num_classes", default=16, type=int)
parser.add_argument("--num_features", default=16, type=int)
parser.add_argument("--num_samples", default=16, type=int)
parser.add_argument("--num_gpus", default=4, type=int)
parser.add_argument("--models_per_gpu", default=4, type=int)
parser.add_argument("--data_dir", default='./', type=str)
parser.add_argument("--model_dir", default='./', type=str)
parser.add_argument("--result_dir", default='./', type=str)
parser.add_argument("--large_data_batch", default=16, type=int)
args = parser.parse_args()
iter_size = args.large_data_batch*args.batch_size
count = 0
hidden_dim_1 = 500
hidden_dim_2 = 500
"""
Change models per gpu for large output spaces
"""

models_per_gpu = args.models_per_gpu
num_gpus=args.num_gpus
begin_time = time.time()

if not args.gpu == 'all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print(os.environ['CUDA_VISIBLE_DEVICES'])

# Training Params
feature_dim = args.num_features

B = args.B
batch_size = args.batch_size
epoch = args.epoch

R = args.R

num_classes = args.num_classes

lookup = np.empty([R, num_classes], dtype=int)
for r in range(R):
    lookup[r] = np.load(os.path.join(args.result_dir, 'b_' +
                                     str(B)+'/lookups/bucket_order_'+str(r)+'.npy'))

params = [None for r in range(R)]
for r in range(R):
    params[r] = np.load(os.path.join(args.model_dir, 'b_' +
                                     str(B)+'/r_'+str(r)+'_epoch_'+epoch+'.npz'))

x_idxs = tf.placeholder(tf.int64, shape=[None, 2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size, feature_dim])

W1_tmp = np.array([params[r]['weights_1'] for r in range(R)])
b1_tmp = np.array([params[r]['bias_1'] for r in range(R)])
W2_tmp = np.array([params[r]['weights_2'] for r in range(R)])
b2_tmp = np.array([params[r]['bias_2'] for r in range(R)])
W3_tmp = np.array([params[r]['weights_3'] for r in range(R)])
b3_tmp = np.array([params[r]['bias_3'] for r in range(R)])

lim_range = R//(models_per_gpu * num_gpus)
weight_size = models_per_gpu * num_gpus
W1 = [None for i in range(weight_size)]
b1 = [None for i in range(weight_size)]
layer_1 = [None for i in range(weight_size)]
W2 = [None for i in range(weight_size)]
b2 = [None for i in range(weight_size)]
layer_2 = [None for i in range(weight_size)]
W3 = [None for i in range(weight_size)]
b3 = [None for i in range(weight_size)]
logits = [None for i in range(weight_size)]
probs = [None for i in range(weight_size)]

lines = []

_midx = 0
for g in range(num_gpus):
    for g_ in range(models_per_gpu):
        ridx = _midx*models_per_gpu*num_gpus + g * models_per_gpu + g_
        with tf.device('/gpu:'+str(g)):
            W1[ridx % weight_size] = tf.Variable(W1_tmp[ridx])
            b1[ridx % weight_size] = tf.Variable(b1_tmp[ridx])
            layer_1[ridx % weight_size] = tf.nn.relu(
                tf.sparse_tensor_dense_matmul(
                    x, W1[ridx % weight_size])+b1[ridx % weight_size])
            #
            W2[ridx % weight_size] = tf.Variable(W2_tmp[ridx])
            b2[ridx % weight_size] = tf.Variable(b2_tmp[ridx])
            layer_2[ridx % weight_size] = tf.nn.relu(tf.matmul(
                layer_1[ridx % weight_size],
                W2[ridx % weight_size])+b2[ridx % weight_size])
            #
            W3[ridx % weight_size] = tf.Variable(W3_tmp[ridx])
            b3[ridx % weight_size] = tf.Variable(b3_tmp[ridx])
            logits[ridx % weight_size] = tf.matmul(
                layer_2[ridx % weight_size],
                W3[ridx % weight_size])+b3[ridx % weight_size]

config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

def _process(_midx):

    ###########################################################
    output = []
    # header = f.readline()
    idxs = []
    vals = []
    count = 0
    offset = 0
    _batch_size = 0
    for line in lines:
        itms = line.strip().split()
        idxs += [(count-offset, int(itm.split(':')[0])) for itm in itms[1:]]
        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
        count += 1
        _batch_size += 1
        if count % batch_size == 0:
            preds = np.reshape(sess.run(
                logits,
                feed_dict={x_idxs: idxs,
                           x_vals: vals}),
                [weight_size, _batch_size, B])
            preds = np.transpose(preds, (1, 0, 2))
            output.append(preds)
            idxs = []
            vals = []
            offset = count
            _batch_size = 0
    return np.vstack(output)


candidates = np.array(range(num_classes))
candidate_indices = np.ascontiguousarray(lookup[:, candidates])
with open(os.path.join(args.data_dir, 'test.txt'), 'r', encoding='utf-8') as fin:
    num_datapoints = int(fin.readline().split()[0])
    score_mat = sp.lil_matrix(
        (num_datapoints, args.num_classes), dtype=np.float32)
    count = 0
    for _i in range(math.ceil(num_datapoints / iter_size)):
        lines = []
        interval = min(iter_size, num_datapoints - (iter_size * _i))
        for i in range(interval):
            lines.append(fin.readline().strip())
        if len(lines) < batch_size:
            break
        preds = []
        for _midx in range(lim_range):
            # with Pool(1) as p:
                # preds.extend(p.map(_process, (_midx,)))
            preds.append(_process(_midx))
        preds = np.ascontiguousarray(np.concatenate(preds, axis=1))
        global_count = preds.shape[0]
        scores = np.zeros((global_count, num_classes), dtype=np.float32)
        gather_batch(preds, candidate_indices, scores, R, B,
                     num_classes, global_count, args.n_threads)
        del preds
        idx = np.arange(global_count).reshape(-1, 1)
        top_lbls_100 = np.argpartition(scores, -100, axis=-1)[:, -100:]
        score_mat[idx + count, top_lbls_100] = scores[idx, top_lbls_100]
        print("Batch [%d]/[%d]" % (_i, math.ceil(num_datapoints / iter_size)))
        count += global_count
    sp.save_npz(os.path.join(args.result_dir, 'score.npz'), score_mat.tocsr())

total_time = time.time()-begin_time
print('overall time_elapsed: ', total_time)
print('pred time (MSec): ', total_time*100/args.num_samples)
train_time = 0
model_size = 0
for i in range(0, R):
    with open(os.path.join(args.result_dir, "log_train_rep_%d.txt" % (i)), 'r') as f:
        lines = f.readlines()
        train_time += float(lines[-2].strip().split(' ')[-1])
        model_size += float(lines[-1].strip().split(' ')[-1])
print('train time (Sec)', train_time)
print('Model Size (MB)', model_size)
