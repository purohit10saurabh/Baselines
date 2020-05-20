from __future__ import print_function
import os
import argparse
from scipy.sparse import csr_matrix
import numpy as np
import timeit
import data_helpers
from scipy.io import savemat
import scipy.sparse as sp

from cnn_model import CNN_model


def _map_rows(X, mapping, shape):
    """Indices should not be repeated
    Will not convert to dense
    """
    X = X.tocsr()  # Avoid this?
    row_idx, col_idx = X.nonzero()
    vals = np.array(X[row_idx, col_idx]).squeeze()
    row_indices = list(map(lambda x: mapping[x], row_idx))
    return csr_matrix(
        (vals, (np.array(row_indices), np.array(col_idx))), shape=shape)


def compute_inv_propesity(labels, A, B):
    num_instances, _ = labels.shape
    freqs = np.ravel(np.sum(labels, axis=0))
    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(freqs+B, -A)
    return np.ravel(wts)


def write_sparse_file(labels, filename, header=True):
    '''
        Write sparse label matrix to text file (comma separated)
        Header: (#users, #labels)
        Args:
            labels: sparse matrix: labels
            filename: str: output file
            header: bool: include header or not
    '''
    if not isinstance(labels, csr_matrix):
        labels = labels.tocsr()
    with open(filename, 'w') as f:
        if header:
            f.write("%d %d\n" % (labels.shape[0], labels.shape[1]))
        for y in labels:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x, v)
                                 for x, v in zip(idx, val)])
            f.write(sentence+'\n')


def load_data(args):
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, train_rows, test_rows = data_helpers.load_data(
        args.data_path, max_length=args.sequence_length, vocab_size=args.vocab_size, labels=args.labels)
    X_trn = X_trn.astype(np.int32)
    X_tst = X_tst.astype(np.int32)
    Y_trn = Y_trn.astype(np.int32)
    Y_tst = Y_tst.astype(np.int32)
    return X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, train_rows, test_rows


def gen_model_file(args):
    data_name = args.data_path.split('/')[-2]
    fs_string = '-'.join([str(fs) for fs in args.filter_sizes])
    file_name = 'data-%s_sl-%d_ed-%d_fs-%s_nf-%d_pu-%d_pt-%s_hd-%d_bs-%d_model-%s_pretrain-%s' % \
        (data_name, args.sequence_length, args.embedding_dim,
         fs_string, args.num_filters, args.pooling_units,
         args.pooling_type, args.hidden_dims, args.batch_size,
         args.model_variation, args.pretrain_type)
    return file_name


def main(args):
    if args.mode == 'show_params':
        print(os.path.join(args.model_dir, gen_model_file(args)))
        model = CNN_model(args)
        model.model_file = os.path.join(args.model_dir, gen_model_file(args))
        model.show_params(args.epoch_idx)
        exit(0)
    print('-'*50)
    print('Loading data...')
    start_time = timeit.default_timer()
    X_trn, Y_trn, X_tst, Y_tst, vocabulary, vocabulary_inv, train_rows, test_rows = load_data(
        args)
    print('Process time %.3f (secs)\n' % (timeit.default_timer() - start_time))

    # Building model
    # ==================================================
    print('-'*50)
    print("Building model...")
    start_time = timeit.default_timer()
    if args.mode == 'train':
        print("Training begins")
        prediction_time = 0
        model = CNN_model(args)
        model.model_file = os.path.join(args.model_dir, gen_model_file(args))
        model.add_data(X_trn, Y_trn, args.labels)
        model.add_pretrain(vocabulary, vocabulary_inv, args.w2v_model_dir)
        model.build_train()

        if not os.path.isdir(model.model_file):
            os.makedirs(model.model_file)
        else:
            print('Warning: model file already exist!\n %s' %
                  (model.model_file))

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)

        f = open(args.results_dir+'/log_train.txt', 'w')

        print('Process time %.3f (secs)\n' %
              (timeit.default_timer() - start_time))
        f.write('Process time %.3f (secs)\n' %
                (timeit.default_timer() - start_time))
        # Training model
        # ==================================================
        file = args.results_dir+'/score.mat'
        print('-'*50)
        f.write('-'*50+'\n')
        print("Training model...")
        start_time = timeit.default_timer()
        f.write("Training model...\n")
        store_params_time = 0.0
        for epoch_idx in range(args.num_epochs + 1):
            loss = model.train()
            print('Iter:', epoch_idx, 'Trn loss ', loss)
            f.write("Iter: %d Trn loss %f\n" % (epoch_idx, loss))
            if (epoch_idx) % 5 == 0:
                print('saving model...')
                tmp_time = timeit.default_timer()
                model.store_params(epoch_idx)
                f.write("%d\n" % epoch_idx)
                store_params_time += timeit.default_timer() - tmp_time
                rest_time = timeit.default_timer() - tmp_time
                tmp_time = timeit.default_timer()
        total_time = timeit.default_timer() - start_time
        print('Total time %.4f (secs), training time %.4f (secs), IO time %.4f (secs), predicition_time (secs) %0.4f'
              % (total_time, total_time - store_params_time-prediction_time-rest_time, store_params_time, prediction_time))
        f.write('Total time %.4f (secs), training time %.4f (secs), IO time %.4f (secs), predicition_time (secs) %0.4f\n'
                % (total_time, total_time - store_params_time-prediction_time-rest_time, store_params_time, prediction_time))

    if args.mode == 'predict':
        print(args)
        print("Predicting")
        prediction_time = 0
        model = CNN_model(args)
        model.model_file = os.path.join(args.model_dir, gen_model_file(args))
        model.add_data(X_tst, Y_tst, args.labels)
        model.add_pretrain(vocabulary, vocabulary_inv, args.w2v_model_dir)
        model.build_predict(args.epoch_idx)
        if not os.path.isdir(model.model_file):
            os.makedirs(model.model_file)
        else:
            print('Warning: model file already exist!\n %s' %
                  (model.model_file))

        if not os.path.isdir(args.results_dir):
            os.makedirs(args.results_dir)
        f = open(args.results_dir+'/log_predict.txt', 'a')
        print('Process time %.3f (secs)\n' %
              (timeit.default_timer() - start_time))
        f.write('Process time %.3f (secs)\n' %
                (timeit.default_timer() - start_time))
        file = args.results_dir+'/score.txt'
        print('-'*50)
        f.write('-'*50+'\n')
        print("Predicting model...")
        start_time = timeit.default_timer()
        f.write("Predicting model...\n")
        tmp_time = timeit.default_timer()
        Yh_tst = model.predict(X_tst, top_k=300)
        prediction_time += timeit.default_timer() - tmp_time
        print("Prediction time per sample: %f" %
              (prediction_time*1000/Yh_tst.shape[0]))
        f.write("Prediction time per sample: %f" %
                (prediction_time*1000/Yh_tst.shape[0]))
        score = _map_rows(Yh_tst, test_rows[1:], test_rows[0])
        write_sparse_file(score, file)


if __name__ == '__main__':
    # Parameters
    # ==================================================
    # Model Variations. See Kim Yoon's Convolutional Neural Networks for
    # Sentence Classification, Section 3 for detail.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        help='raw data path in CPickle format', type=str,
                        default='../sample_data/rcv1_raw_small.p')
    parser.add_argument('--sequence_length',
                        help='max sequence length of a document', type=int,
                        default=500)
    parser.add_argument('--embedding_dim',
                        help='dimension of word embedding representation', type=int,
                        default=300)
    parser.add_argument('--filter_sizes',
                        help='number of filter sizes (could be a list of integer)', type=int,
                        default=[2, 4, 8], nargs='+')
    parser.add_argument('--num_filters',
                        help='number of filters (i.e. kernels) in CNN model', type=int,
                        default=32)
    parser.add_argument('--pooling_units',
                        help='number of pooling units in 1D pooling layer', type=int,
                        default=32)
    parser.add_argument('--pooling_type',
                        help='max or average', type=str,
                        default='max')
    parser.add_argument('--hidden_dims',
                        help='number of hidden units', type=int,
                        default=512)
    parser.add_argument('--model_variation',
                        help='model variation: CNN-rand or CNN-pretrain', type=str,
                        default='pretrain')
    parser.add_argument('--pretrain_type',
                        help='pretrain model: GoogleNews or glove', type=str,
                        default='glove')
    parser.add_argument('--batch_size',
                        help='number of batch size', type=int,
                        default=256)
    parser.add_argument('--num_epochs',
                        help='number of epcohs for training', type=int,
                        default=50)
    parser.add_argument('--vocab_size',
                        help='size of vocabulary keeping the most frequent words', type=int,
                        default=50000)
    parser.add_argument('--labels',
                        help='label space', type=int,
                        default=101)
    parser.add_argument('--w2v_model_dir',
                        help='w2v model directory', type=str,
                        default=101)

    parser.add_argument('--model_dir',
                        help='model directory', type=str,
                        default=101)
    parser.add_argument('--results_dir',
                        help='results directory', type=str,
                        default=101)
    parser.add_argument('--mode',
                        help='mode', type=str,
                        default='train')

    parser.add_argument('--epoch_idx',
                        help='Load epoch entry', type=int,
                        default=0)
    args = parser.parse_args()
    main(args)
