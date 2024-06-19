import os
import math
# import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
from utils import *
from torch_geometric.utils import dropout_adj, convert
from torch_sparse import SparseTensor
import torch.nn as nn
import networkx as nx
from multiprocessing import Pool


class FairDataset:
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

    @staticmethod
    def adj2edge_index(adj, fea_num):
        edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        edge_index_spar = SparseTensor.from_edge_index(edge_index, sparse_sizes=(fea_num, fea_num), )
        return edge_index, edge_index_spar

    def load_data(self, sens_attr=None, label_number=None, split_ratio=None, val_idx=True, sens_number=None):
        """
            Load data
        """
        # Load german dataset
        if self.dataset == 'german':
            dataset = 'german'
            sens_attr = "Gender"
            predict_attr = "GoodCustomer"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            path_german = "./datasets/german"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_german(dataset, sens_attr,
                                                                                    predict_attr, path=path_german,
                                                                                    label_number=label_number,
                                                                                    split_ratio=split_ratio,
                                                                                    val_idx=val_idx
                                                                                    )
        # Load bail dataset
        elif self.dataset == 'bail':
            dataset = 'bail'
            sens_attr = "WHITE"  # column number after feature process is 0
            sens_idx = 0
            predict_attr = "RECID"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            path_bail = "./datasets/bail"
            adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail(dataset, sens_attr,
                                                                                  predict_attr, path=path_bail,
                                                                                  label_number=label_number,
                                                                                  split_ratio=split_ratio,
                                                                                  val_idx=val_idx
                                                                                  )
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
        # load pokec dataset
        elif self.dataset == 'pokec_z':
            dataset = 'region_job'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 4000 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./datasets/pokec/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
        elif self.dataset == 'pokec_n':
            dataset = 'region_job_2'
            sens_attr = "region"
            predict_attr = "I_am_working_in_field"
            label_number = 3500 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_number = 200
            sens_idx = 3
            seed = 20
            path = "./datasets/pokec/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
        elif self.dataset == 'nba':
            dataset = 'nba'
            sens_attr = "country"
            predict_attr = "SALARY"
            label_number = 100 if label_number is None else label_number
            sens_number = label_number if sens_number is None else sens_number
            sens_idx = 35
            seed = 20
            path = "./datasets/NBA/"
            adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset,
                                                                                                   sens_attr,
                                                                                                   predict_attr,
                                                                                                   path=path,
                                                                                                   label_number=label_number,
                                                                                                   sens_number=sens_number,
                                                                                                   seed=seed,
                                                                                                   split_ratio=split_ratio,
                                                                                                   val_idx=val_idx)
            labels[labels > 1] = 1
            norm_features = feature_norm(features)
            norm_features[:, sens_idx] = features[:, sens_idx]
            features = norm_features
        else:
            print('Invalid dataset name!!')
            exit(0)

        edge_index, edge_index_spar = FairDataset.adj2edge_index(adj=adj, fea_num=features.shape[0])
        edge_index_spar = edge_index_spar.to(self.device)
        edge_index = edge_index.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        sens = sens.to(self.device)
        self.label_number, self.sens_number = label_number, sens_number
        (self.edge_index, self.edge_index_nor, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test,
         self.sens) = (edge_index_spar, edge_index, features, labels, idx_train, idx_val, idx_test, sens)


def load_pokec(dataset, sens_attr, predict_attr, path="./Dataset/pokec/", label_number=1000, sens_number=500,
               seed=19, split_ratio=None, val_idx=True):
    """Load data"""
    # print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    # seed = 20
    random.seed(seed)
    label_idx = np.where(labels >= 0)[0]
    random.shuffle(label_idx)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = label_idx[:min(int(split_ratio[0] * len(label_idx)), label_number)]
    if val_idx:
        idx_val = label_idx[int(split_ratio[0] * len(label_idx)):int((split_ratio[0] + split_ratio[1]) * len(label_idx))]
        idx_test = label_idx[int((split_ratio[0] + split_ratio[1]) * len(label_idx)):]
    else:
        idx_test = label_idx[label_number:]
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train


def load_bail(dataset, sens_attr="WHITE", predict_attr="RECID", path="./Dataset/bail/", label_number=1000,
              split_ratio=None, val_idx=True):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    # header.remove(sens_attr)

    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = np.append(label_idx_0[:min(int(split_ratio[0] * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(split_ratio[0] * len(label_idx_1)), label_number // 2)])

    if val_idx:
        idx_val = np.append(label_idx_0[int(split_ratio[0] * len(label_idx_0)):int(
            (split_ratio[0] + split_ratio[1]) * len(label_idx_0))],
                            label_idx_1[int(split_ratio[0] * len(label_idx_1)):int(
                                (split_ratio[0] + split_ratio[1]) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((split_ratio[0] + split_ratio[1]) * len(label_idx_0)):],
                             label_idx_1[int((split_ratio[0] + split_ratio[1]) * len(label_idx_1)):])
    else:
        idx_test = np.append(label_idx_0[min(int(split_ratio[0] * len(label_idx_0)), label_number // 2):],
                             label_idx_1[min(int(split_ratio[0] * len(label_idx_1)), label_number // 2):])
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens


def load_german(dataset, sens_attr="Gender", predict_attr="GoodCustomer", path="./Datasets/german/",
                label_number=1000, split_ratio=None, val_idx=True):
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('OtherLoansAtStore')
    header.remove('PurposeOfLoan')

    # Sensitive Attribute
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Female'] = 1
    idx_features_labels['Gender'][idx_features_labels['Gender'] == 'Male'] = 0
   
    # build relationship
    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.8)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[labels == -1] = 0

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    if split_ratio is None:
        split_ratio = [0.5, 0.25, 0.25]

    idx_train = np.append(label_idx_0[:min(int(split_ratio[0] * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(split_ratio[0] * len(label_idx_1)), label_number // 2)])

    if val_idx:
        idx_val = np.append(label_idx_0[int(split_ratio[0] * len(label_idx_0)):int(
            (split_ratio[0] + split_ratio[1]) * len(label_idx_0))],
                            label_idx_1[int(split_ratio[0] * len(label_idx_1)):int(
                                (split_ratio[0] + split_ratio[1]) * len(label_idx_1))])
        idx_test = np.append(label_idx_0[int((split_ratio[0] + split_ratio[1]) * len(label_idx_0)):],
                             label_idx_1[int((split_ratio[0] + split_ratio[1]) * len(label_idx_1)):])
    else:
        idx_test = np.append(label_idx_0[min(int(split_ratio[0] * len(label_idx_0)), label_number // 2):],
                             label_idx_1[min(int(split_ratio[0] * len(label_idx_1)), label_number // 2):])
        idx_val = idx_test

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens
