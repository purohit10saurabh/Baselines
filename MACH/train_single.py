# exit(0)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# tf.logging.set_verbosity('ERROR')
import numpy as np
import glob
import time
from itertools import islice
import argparse
import math
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--repetition", help="which repetition?", default=0)
parser.add_argument("--B", help="How many buckets?", default=2000)
parser.add_argument("--gpu", help="which GPU?", default=0)
parser.add_argument(
    "--gpu_usage", help="how much GPU memory to use", default=0.25)
parser.add_argument("--batch_size", default=100)
parser.add_argument("--n_epochs", default=60)
parser.add_argument("--load_epoch", default=0)
parser.add_argument("--data_dir", type=str, default='./')
parser.add_argument("--result_dir", type=str, default='./')
parser.add_argument("--model_dir", type=str, default='./')
parser.add_argument("--num_features", type=int, default=0)
parser.add_argument("--n_train", type=int, default=0)

args = parser.parse_args()

if not args.gpu == 'all':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# Training Params
feature_dim = args.num_features
n_classes = int(args.B)
hidden_dim_1 = 500
hidden_dim_2 = 500
n_epochs = int(args.n_epochs)
batch_size = int(args.batch_size)
data_path = args.data_dir
model_dir = args.model_dir
result_dir = args.result_dir
r = int(args.repetition)  # which repetition
# will load weights and biases from this epoch number if found in saved_models folder. If not found or if given 0, we do random initialization
load_epoch = int(args.load_epoch)

# load lookups
lookup = np.load(os.path.join(
    result_dir,
    'b_'+str(n_classes)+'/lookups/bucket_order_'+str(r)+'.npy'
))

# check for saved models and create graph
x_idxs = tf.placeholder(tf.int64, shape=[None, 2])
x_vals = tf.placeholder(tf.float32, shape=[None])
x = tf.SparseTensor(x_idxs, x_vals, [batch_size, feature_dim])
y = tf.placeholder(tf.float32, shape=[None, n_classes])

saved_models = glob.glob(os.path.join(
    model_dir, 'b_'+str(n_classes)+'/r_'+str(r)+'_epoch_'+str(load_epoch)+'.npz'))

if not saved_models:
    W1 = tf.Variable(tf.truncated_normal(
        [feature_dim, hidden_dim_1], stddev=0.05))
    b1 = tf.Variable(tf.truncated_normal([hidden_dim_1], stddev=0.05))
    layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x, W1)+b1)
    #
    W2 = tf.Variable(tf.truncated_normal(
        [hidden_dim_1, hidden_dim_2], stddev=0.05))
    b2 = tf.Variable(tf.truncated_normal([hidden_dim_2], stddev=0.05))
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W2)+b2)
    #
    W3 = tf.Variable(tf.truncated_normal(
        [hidden_dim_2, n_classes], stddev=0.05))
    b3 = tf.Variable(tf.truncated_normal([n_classes], stddev=0.05))
    logits = tf.matmul(layer_2, W3)+b3
else:
    params = np.load(saved_models[0])
    #
    W1_tmp = tf.placeholder(tf.float32, shape=[feature_dim, hidden_dim_1])
    b1_tmp = tf.placeholder(tf.float32, shape=[hidden_dim_1])
    W1 = tf.Variable(W1_tmp)
    b1 = tf.Variable(b1_tmp)
    layer_1 = tf.nn.relu(tf.sparse_tensor_dense_matmul(x, W1)+b1)
    #
    W2_tmp = tf.placeholder(tf.float32, shape=[hidden_dim_1, hidden_dim_2])
    b2_tmp = tf.placeholder(tf.float32, shape=[hidden_dim_2])
    W2 = tf.Variable(W2_tmp)
    b2 = tf.Variable(b2_tmp)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, W2)+b2)
    #
    W3_tmp = tf.placeholder(tf.float32, shape=[hidden_dim_2, n_classes])
    b3_tmp = tf.placeholder(tf.float32, shape=[n_classes])
    W3 = tf.Variable(W3_tmp)
    b3 = tf.Variable(b3_tmp)
    logits = tf.matmul(layer_2, W3)+b3

loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

train_step = tf.train.AdamOptimizer().minimize(loss)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_usage)
sess = tf.Session(config=config)
if not saved_models:
    sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer(), feed_dict={W1_tmp: params['weights_1'], b1_tmp: params['bias_1'],
                                                           W2_tmp: params['weights_2'], b2_tmp: params['bias_2'], W3_tmp: params['weights_3'], b3_tmp: params['bias_3']})

# Load data and train


def data_generator(files, batch_size, feature_dim, n_classes):
    while 1:
        lines = []
        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                header = f.readline()
                while True:
                    temp = len(lines)
                    lines += list(islice(f, batch_size-temp))
                    if len(lines) != batch_size:
                        break
                    idxs = []
                    vals = []
                    ##
                    y_idxs = []
                    y_vals = []
                    y_batch = np.zeros([batch_size, n_classes], dtype=float)
                    count = 0
                    for line in lines:
                        itms = line.strip().split()
                        ##
                        try:
                            y_idxs = [int(itm) for itm in itms[0].split(',')]
                        except Exception as e:
                            print(line, "\n", itms)

                        y_vals = [1.0 for itm in range(len(y_idxs))]
                        for i in range(len(y_idxs)):
                            y_batch[count, lookup[y_idxs[i]]] = y_vals[i]
                        ##
                        idxs += [(count, int(itm.split(':')[0]))
                                 for itm in itms[1:]]
                        vals += [float(itm.split(':')[1]) for itm in itms[1:]]
                        count += 1
                    lines = []
                    yield (idxs, vals, y_batch)


train_files = glob.glob(os.path.join(data_path, 'train.txt'))

training_data_generator = data_generator(
    train_files, batch_size, feature_dim, n_classes)

curr_epoch = load_epoch
old_saved_epoch = -1
n_train = args.n_train
n_steps_per_epoch = n_train//batch_size
n_steps = n_epochs*n_steps_per_epoch

print('******************************************')
print('n_steps:', n_steps)
print('******************************************')

prog_begin_time = time.time()
begin_time = time.time()
last_addr=''
n_check = 100
for i in range(n_steps):
    idxs_batch, vals_batch, labels_batch = next(training_data_generator)
    sess.run(train_step, feed_dict={
             x_idxs: idxs_batch, x_vals: vals_batch, y: labels_batch})
    #
    if i % n_check == 0:
        print('Finished ', i, ' steps. Time elapsed for last ' +
              str(n_check)+' batches = ', time.time()-begin_time)
        begin_time = time.time()
        train_loss = sess.run(
            loss, feed_dict={x_idxs: idxs_batch, x_vals: vals_batch, y: labels_batch})
        print('train loss: ', train_loss)
        print('#######################')
    if i % n_steps_per_epoch == 0 and i > 0:
        curr_epoch += 1
        if curr_epoch % 5 == 0:
            params = sess.run([W1, b1, W2, b2, W3, b3])
            last_addr=os.path.join(
                                model_dir,
                                'b_'+str(n_classes)+'/r_'+str(r)+'_epoch_'+str(curr_epoch)+'.npz')
            np.savez_compressed(last_addr,
                weights_1=params[0], bias_1=params[1], 
                weights_2=params[2], bias_2=params[3], 
                weights_3=params[4], bias_3=params[5])
            del params
            if old_saved_epoch > -1:
                os.remove(os.path.join(
                    model_dir,
                    'b_'+str(n_classes)+'/r_'+str(r)+'_epoch_'+str(old_saved_epoch)+'.npz'))
            old_saved_epoch = curr_epoch

if i % n_steps_per_epoch != 0:
    curr_epoch += 1
    if curr_epoch % 5 == 0:
        params = sess.run([W1, b1, W2, b2, W3, b3])
        last_addr=os.path.join(
                                model_dir,
                                'b_'+str(n_classes)+'/r_'+str(r)+'_epoch_'+str(curr_epoch)+'.npz')
        np.savez_compressed(last_addr,
            weights_1=params[0], bias_1=params[1], 
            weights_2=params[2], bias_2=params[3], 
            weights_3=params[4], bias_3=params[5])
        os.remove(os.path.join(
            model_dir,
            'b_'+str(n_classes)+'/r_'+str(r)+'_epoch_'+str(old_saved_epoch)+'.npz'))
        old_saved_epoch = curr_epoch
        del params
print("Training Completed at (Sec)", time.time()-prog_begin_time)
print("Model Size (MB)", os.path.getsize(last_addr)/math.pow(2, 20))
