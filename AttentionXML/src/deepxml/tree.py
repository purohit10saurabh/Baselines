#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2019/2/26
@author yrh

"""

import os
import time
import click
import numpy as np
import torch
from pathlib import Path
from ruamel.yaml import YAML
from multiprocessing import Process
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from logzero import logger
from xclib.data import data_utils as du
from deepxml.data_utils import get_data, get_mlb, get_word_emb, output_res
from deepxml.dataset import MultiLabelDataset, XMLDataset
from deepxml.models import Model, XMLModel
from deepxml.cluster import build_tree_by_level
from deepxml.networks import *


class FastAttentionXML(object):
    """

    """
    def __init__(self, labels_num, data_cnf, model_cnf, tree_id=''):
        self.data_cnf, self.model_cnf = data_cnf, model_cnf
        self.model_path = os.path.join(model_cnf['path'], F'{model_cnf["name"]}-{data_cnf["name"]}{tree_id}')
        self.emb_init, self.level = get_word_emb(data_cnf['embedding']['emb_init']), model_cnf['level']
        self.labels_num, self.models = labels_num, {}
        self.inter_group_size, self.top = model_cnf['k'], model_cnf['top']
        self.groups_path = os.path.join(model_cnf['path'], F'{model_cnf["name"]}-{data_cnf["name"]}{tree_id}-cluster')

    @staticmethod
    def get_mapping_y(groups, labels_num, *args):
        mapping = np.empty(labels_num + 1, dtype=np.long)
        for idx, labels_list in enumerate(groups):
            mapping[labels_list] = idx
        mapping[labels_num] = len(groups)
        return (FastAttentionXML.get_group_y(mapping, y, len(groups)) for y in args)

    @staticmethod
    def get_group_y(mapping: np.ndarray, data_y: csr_matrix, num_groups):
        r, c, d = [], [], []
        for i in range(data_y.shape[0]):
            g = np.unique(mapping[data_y.indices[data_y.indptr[i]: data_y.indptr[i + 1]]])
            r += [i] * len(g)
            c += g.tolist()
            d += [1] * len(g)
        return csr_matrix((d, (r, c)), shape=(data_y.shape[0], num_groups))

    def train_level(self, level, train_x, train_y, valid_x, valid_y):
        model_cnf, data_cnf = self.model_cnf, self.data_cnf
        if level == 0:
            while not os.path.exists(F'{self.groups_path}-Level-{level}.npy'):
                time.sleep(30)
            groups = np.load(F'{self.groups_path}-Level-{level}.npy', allow_pickle=True)
            train_y, valid_y = self.get_mapping_y(groups, self.labels_num, train_y, valid_y)
            labels_num = len(groups)
            train_loader = DataLoader(MultiLabelDataset(train_x, train_y),
                                      model_cnf['train'][level]['batch_size'], num_workers=4, shuffle=True)
            valid_loader = DataLoader(MultiLabelDataset(valid_x, valid_y, training=False),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = Model(AttentionRNN, labels_num=labels_num, model_path=F'{self.model_path}-Level-{level}',
                          emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            if not os.path.exists(model.model_path):
                logger.info(F'Training Level-{level}, Number of Labels: {labels_num}')
                model.train(train_loader, valid_loader, **model_cnf['train'][level])
                model.optimizer = None
                logger.info(F'Finish Training Level-{level}')
            self.models[level] = model
            logger.info(F'Generating Candidates for Level-{level+1}, '
                        F'Number of Labels: {labels_num}, Top: {self.top}')
            train_loader = DataLoader(MultiLabelDataset(train_x), model_cnf['valid']['batch_size'], num_workers=4)
            return train_y, model.predict(train_loader, k=self.top), model.predict(valid_loader, k=self.top)
        else:
            train_group_y, train_group, valid_group = self.train_level(level - 1, train_x, train_y, valid_x, valid_y)
            torch.cuda.empty_cache()

            logger.info('Getting Candidates')
            _, group_labels = train_group
            group_candidates = np.empty((len(train_x), self.top), dtype=np.int)
            for i, labels in tqdm(enumerate(group_labels), leave=False, desc='Generating'):
                ys, ye = train_group_y.indptr[i], train_group_y.indptr[i + 1]
                positive = set(train_group_y.indices[ys: ye])
                if self.top >= len(positive):
                    candidates = positive
                    for la in labels:
                        if len(candidates) == self.top:
                            break
                        if la not in candidates:
                            candidates.add(la)
                else:
                    candidates = set()
                    for la in labels:
                        if la in positive:
                            candidates.add(la)
                        if len(candidates) == self.top:
                            break
                    if len(candidates) < self.top:
                        candidates = (list(candidates) + list(positive - candidates))[:self.top]
                group_candidates[i] = np.asarray(list(candidates))

            if level < self.level - 1:
                while not os.path.exists(F'{self.groups_path}-Level-{level}.npy'):
                    time.sleep(30)
                groups = np.load(F'{self.groups_path}-Level-{level}.npy', allow_pickle=True)
                train_y, valid_y = self.get_mapping_y(groups, self.labels_num, train_y, valid_y)
                labels_num, last_groups = len(groups), self.get_inter_groups(len(groups))
            else:
                groups, labels_num = None, train_y.shape[1]
                last_groups = np.load(F'{self.groups_path}-Level-{level-1}.npy', allow_pickle=True)

            train_loader = DataLoader(XMLDataset(train_x, train_y, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_candidates),
                                      model_cnf['train'][level]['batch_size'], num_workers=4, shuffle=True)
            group_scores, group_labels = valid_group
            valid_loader = DataLoader(XMLDataset(valid_x, valid_y, training=False, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_labels,
                                                 group_scores=group_scores),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            model = XMLModel(network=FastAttentionRNN, labels_num=labels_num, emb_init=self.emb_init,
                             model_path=F'{self.model_path}-Level-{level}', **data_cnf['model'], **model_cnf['model'])
            if not os.path.exists(model.model_path):
                logger.info(F'Loading parameters of Level-{level} from Level-{level-1}')
                last_model = self.get_last_models(level - 1)
                model.network.module.emb.load_state_dict(last_model.module.emb.state_dict())
                model.network.module.lstm.load_state_dict(last_model.module.lstm.state_dict())
                model.network.module.linear.load_state_dict(last_model.module.linear.state_dict())
                logger.info(F'Training Level-{level}, '
                            F'Number of Labels: {labels_num}, '
                            F'Candidates Number: {train_loader.dataset.candidates_num}')
                model.train(train_loader, valid_loader, **model_cnf['train'][level])
                model.optimizer = model.state = None
                logger.info(F'Finish Training Level-{level}')
            self.models[level] = model
            if level == self.level - 1:
                return
            logger.info(F'Generating Candidates for Level-{level+1}, '
                        F'Number of Labels: {labels_num}, Top: {self.top}')
            group_scores, group_labels = train_group
            train_loader = DataLoader(XMLDataset(train_x, labels_num=labels_num,
                                                 groups=last_groups, group_labels=group_labels,
                                                 group_scores=group_scores),
                                      model_cnf['valid']['batch_size'], num_workers=4)
            return train_y, model.predict(train_loader, k=self.top), model.predict(valid_loader, k=self.top)

    def get_last_models(self, level):
        return self.models[level].model if level == 0 else self.models[level].network

    def predict_level(self, level, test_x, k, labels_num):
        data_cnf, model_cnf = self.data_cnf, self.model_cnf
        model = self.models.get(level, None)
        if level == 0:
            logger.info(F'Predicting Level-{level}, Top: {k}')
            if model is None:
                model = Model(AttentionRNN, labels_num=labels_num, model_path=F'{self.model_path}-Level-{level}',
                              emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            test_loader = DataLoader(MultiLabelDataset(test_x), model_cnf['predict']['batch_size'],
                                     num_workers=4)
            return model.predict(test_loader, k=k)
        else:
            if level == self.level - 1:
                groups = np.load(F'{self.groups_path}-Level-{level-1}.npy', allow_pickle=True)
            else:
                groups = self.get_inter_groups(labels_num)
            group_scores, group_labels = self.predict_level(level - 1, test_x, self.top, len(groups))
            torch.cuda.empty_cache()
            logger.info(F'Predicting Level-{level}, Top: {k}')
            if model is None:
                model = XMLModel(network=FastAttentionRNN, labels_num=labels_num,
                                 model_path=F'{self.model_path}-Level-{level}',
                                 emb_init=self.emb_init, **data_cnf['model'], **model_cnf['model'])
            test_loader = DataLoader(XMLDataset(test_x, labels_num=labels_num,
                                                groups=groups, group_labels=group_labels, group_scores=group_scores),
                                     model_cnf['predict']['batch_size'], num_workers=4)
            return model.predict(test_loader, k=k)

    def get_inter_groups(self, labels_num):
        assert labels_num % self.inter_group_size == 0
        return np.asarray([list(range(i, i + self.inter_group_size))
                           for i in range(0, labels_num, self.inter_group_size)])

    def train(self, train_x, train_y, valid_x, valid_y):
        self.train_level(self.level - 1, train_x, train_y, valid_x, valid_y)

    def predict(self, test_x, k=100):
        return self.predict_level(self.level - 1, test_x, k, self.labels_num)


@click.command()
@click.option('-d', '--data-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('-m', '--model-cnf', type=click.Path(exists=True), help='Path of dataset configure yaml.')
@click.option('--mode', type=click.Choice(['train', 'eval']), default=None)
@click.option('-t', '--tree-id', type=click.INT, default=None)
@click.option('-lbs', '--num_labels', type=click.INT, default=0)
def main(data_cnf, model_cnf, mode, tree_id=None, num_labels=0):
    tree_id = F'-Tree-{tree_id}' if tree_id is not None else ''
    yaml = YAML(typ='safe')
    data_cnf, model_cnf = yaml.load(Path(data_cnf)), yaml.load(Path(model_cnf))
    model, model_name = None, model_cnf['name']

    if mode is None or mode == 'train':
        logger.info('Loading Training and Validation Set')
        train_x, train_labels = get_data(data_cnf['train']['texts'], data_cnf['train']['labels'])
        if 'size' in data_cnf['valid']:
            random_state = data_cnf['valid'].get('random_state', 1240)
            train_x, valid_x, train_labels, valid_labels = train_test_split(train_x, train_labels,
                                                                            test_size=data_cnf['valid']['size'],
                                                                            random_state=random_state)
        else:
            valid_x, valid_labels = get_data(data_cnf['valid']['texts'], data_cnf['valid']['labels'])
        mlb = get_mlb(data_cnf['labels_binarizer'], np.hstack((train_labels, valid_labels)))
        train_y, valid_y = mlb.transform(train_labels), mlb.transform(valid_labels)
        labels_num = len(mlb.classes_)
        logger.info(F'Number of Labels: {labels_num}')
        logger.info(F'Size of Training Set: {len(train_x)}')
        logger.info(F'Size of Validation Set: {len(valid_x)}')

        logger.info('Training')
        model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
        model_cnf['cluster']['groups_path'] = model.groups_path
        cluster_process = Process(target=build_tree_by_level,
                                  args=(data_cnf['train']['sparse'], mlb),
                                  kwargs=model_cnf['cluster'])
        cluster_process.start()
        start_time = time.time()
        model.train(train_x, train_y, valid_x, valid_y)
        total_time = time.time() - start_time
        print("Total Training Time: {} Sec".format(total_time))
        cluster_process.join()
        logger.info('Finish Training')

    if mode is None or mode == 'eval':
        print(num_labels)
        logger.info('Loading Test Set')
        mlb = get_mlb(data_cnf['labels_binarizer'])
        labels_num = len(mlb.classes_)
        test_x, _ = get_data(data_cnf['test']['texts'], None)
        logger.info(F'Size of Test Set: {len(test_x)}')

        if model is None:
            model = FastAttentionXML(labels_num, data_cnf, model_cnf, tree_id)
        logger.info('Predicting')
        start_time = time.time()
        scores, labels = model.predict(test_x, k=100)
        total_time = time.time() - start_time
        logger.info('Finish Predicting')
        labels = mlb.classes_[labels]
        output_res(data_cnf['output']['res'], F'{model_name}-{data_cnf["test"]["name"]}{tree_id}', scores, labels)
        print("Prediction Time per samples: {0:.2f} MSec".format(total_time*1000/labels.shape[0]))
        _scores = lil_matrix((len(scores), num_labels), dtype=np.float32)
        batch_size = 100000
        for idx in range(0, len(scores), batch_size):
            start = idx
            end = min(len(scores), start+batch_size)
            indexes = np.arange(start, end).reshape(-1, 1)
            _scores[indexes, labels[start:end]] = scores[start:end]
        du.write_sparse_file(_scores.tocsr(), os.path.join(data_cnf['output']['res'], F"score.txt{tree_id}"))


if __name__ == '__main__':
    main()
