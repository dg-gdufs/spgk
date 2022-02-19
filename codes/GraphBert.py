'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 00:04:32
LastEditors: Renhetian
LastEditTime: 2022-02-19 01:16:27
'''

import os
import pickle
import torch
import hashlib
import numpy as np
import scipy.sparse as sp
from numpy.linalg import inv


class GraphBert:
    
    load_path = 'data/graphbert_dataset/'
    save_path = 'data/graphbert_loader/'

    def __init__(self, dataset_name='default') -> None:
        self.dataset_name = dataset_name
        self.load_path += dataset_name
        self.save_path += dataset_name

        self.graphbert_data = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path)

    def save(self, data, save_name):
        save_path = self.save_path + '/{}.pkl'.format(save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def encode_onehot(self, labels):
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def build_graphbert_data(self, c=0.15):
        idx_features_labels = np.genfromtxt("{}/node".format(self.load_path), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        one_hot_labels = self.encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        index_id_map = {i: j for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}/link".format(self.load_path),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(one_hot_labels.shape[0], one_hot_labels.shape[0]),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * self.adj_normalize(adj)).toarray())

        norm_adj = self.adj_normalize(adj + sp.eye(adj.shape[0]))

        idx_train = range(9000)
        idx_val = range(9000, 10000)
        idx_test = range(9000, 10000)

        features = torch.FloatTensor(np.array(features.todense()))
        labels = torch.LongTensor(np.where(one_hot_labels)[1])
        adj = self.sparse_mx_to_torch_sparse_tensor(norm_adj)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        # if self.load_all_tag:
        #     hop_dict, wl_dict, batch_dict = self.load_hop_wl_batch()
        #     raw_feature_list = []
        #     role_ids_list = []
        #     position_ids_list = []
        #     hop_ids_list = []
        #     for node in idx:
        #         node_index = idx_map[node]
        #         neighbors_list = batch_dict[node]

        #         raw_feature = [features[node_index].tolist()]
        #         role_ids = [wl_dict[node]]
        #         position_ids = range(len(neighbors_list) + 1)
        #         hop_ids = [0]
        #         for neighbor, intimacy_score in neighbors_list:
        #             neighbor_index = idx_map[neighbor]
        #             raw_feature.append(features[neighbor_index].tolist())
        #             role_ids.append(wl_dict[neighbor])
        #             if neighbor in hop_dict[node]:
        #                 hop_ids.append(hop_dict[node][neighbor])
        #             else:
        #                 hop_ids.append(99)
        #         raw_feature_list.append(raw_feature)
        #         role_ids_list.append(role_ids)
        #         position_ids_list.append(position_ids)
        #         hop_ids_list.append(hop_ids)
        #     raw_embeddings = torch.FloatTensor(raw_feature_list)
        #     wl_embedding = torch.LongTensor(role_ids_list)
        #     hop_embeddings = torch.LongTensor(hop_ids_list)
        #     int_embeddings = torch.LongTensor(position_ids_list)
        # else:
        raw_embeddings, wl_embedding, hop_embeddings, int_embeddings = None, None, None, None

        self.graphbert_data = {'X': features, 'A': adj, 'S': eigen_adj, 'index_id_map': index_id_map, 'edges': edges_unordered, 'raw_embeddings': raw_embeddings, 'wl_embedding': wl_embedding, 'hop_embeddings': hop_embeddings, 'int_embeddings': int_embeddings, 'y': labels, 'idx': idx, 'idx_train': idx_train, 'idx_test': idx_test, 'idx_val': idx_val}
        self.save(self.graphbert_data, 'graphbert_data')

    def build_wl(self, max_iter=2):
        if not self.graphbert_data:
            print("empty graphbert_data")
            return 

        node_color_dict = {}
        node_neighbor_dict = {}
        node_list = self.graphbert_data['idx']
        link_list = self.graphbert_data['edges']

        for node in node_list:
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}
        for pair in link_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1

        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = node_neighbor_dict[node]
                neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
                color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing
            color_index_dict = {k: v+1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]
            if node_color_dict == new_color_dict or iteration_count == max_iter:
                break
            else:
                node_color_dict = new_color_dict
            iteration_count += 1

        self.save(node_color_dict, 'wl_'+str(max_iter))

    def build_batch(self, k=5):
        pass
