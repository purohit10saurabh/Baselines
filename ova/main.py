import sys
import os
import numpy as np
import xclib.data.data_loader as data_loader
import parameters as parameters
import ova
import xclib.utils.utils as utils
import time
from xclib.data import data_utils
import scipy.sparse as sparse
import _pickle as pickle


def _get_fname(params):
    os.makedirs(params.model_dir, exist_ok=True)
    _ext = ""
    if params.start_index != 0 or params.end_index != -1:
        _ext = "_{}_{}".format(params.start_index, params.end_index)
    model_fname = os.path.join(params.model_dir, params.model_fname+_ext)
    return model_fname


def create_classifier(params):
    if params.clf_type == 'ova':
        return ova.OVAClassifier(solver='liblinear',
                                 loss='squared_hinge',
                                 C=params.C,
                                 tol=params.tol,
                                 verbose=0,
                                 batch_size=params.batch_size,
                                 norm=params.norm,
                                 num_threads=params.num_threads,
                                 max_iter=params.max_iter,
                                 threshold=params.threshold,
                                 feature_type=params.feature_type,
                                 dual=params.dual)
    else:
        raise NotImplementedError("Unknown classifier!")


def main(args):
    clf = create_classifier(args.params)
    args.save('train_parameters.json')
    if args.params.mode == 'train':
        model_fname = _get_fname(args.params)
        clf.fit(data_dir=args.params.data_dir,
                dataset=args.params.dataset,
                model_dir=args.params.model_dir,
                feat_fname=args.params.tr_feat_fname,
                label_fname=args.params.tr_label_fname,
                save_after=2000)
        clf.save(model_fname)
    elif args.params.mode == 'predict':
        clf.load(os.path.join(args.params.model_dir, args.params.model_fname))
        predicted_labels = clf.predict(
            data_dir=args.params.data_dir,
            dataset=args.params.dataset,
            feat_fname=args.params.ts_feat_fname,
            label_fname=args.params.ts_label_fname)
        data_utils.write_sparse_file(
            predicted_labels,
            os.path.join(args.params.result_dir, 'score.txt'))
    else:
        raise NotImplementedError("Mode not implemented!")


if __name__ == '__main__':
    args = parameters.Parameters("Parameters")
    #  Load parameters from a json file
    # if len(sys.argv) > 1:
    #     args.load(sys.argv[1])
    #  Parse arguments; Overwrite with command line arguments if required
    args.parse_args()
    main(args)
