'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-26 21:02:11
LastEditors: Renhetian
LastEditTime: 2022-01-27 21:17:27
'''

import os
import torch
import hashlib
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from codes.Utils import doc_format, spgk, device


class Preprocess:

    def __init__(self, loader) -> None:
        self.loader = loader

    def load_file(self, file_path):
        for file_name in os.listdir(file_path):
            if file_name.endswith('.neg'):
                with open(file_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    for i in file:
                        self.loader.data.append(doc_format(i))
                        self.loader.label.append(0)
            elif file_name.endswith('.pos'):
                with open(file_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    for i in file:
                        self.loader.data.append(doc_format(i))
                        self.loader.label.append(1)
        self.loader.label = np.array(self.loader.label)
        self.loader.save([self.loader.data, self.loader.label], 'data')

    def build_kernel_matrix(self, window_size=2, depth=1):
        '''
        Build kernel matrices.
        构建图核。
        '''
        if not self.loader.data:
            print("empty data")
            return 

        graphs = list()
        sizes = list()
        degs = list()
        print("\nGraphs of words building progress:")
        for doc in tqdm(self.loader.data):
            doc = doc.split()
            G = nx.Graph()
            for i in range(len(doc)):
                if doc[i] not in G.nodes():
                    G.add_node(doc[i])
                for j in range(i+1, i+window_size):
                    if j < len(doc):
                        G.add_edge(doc[i], doc[j])
            graphs.append(G)
            sizes.append(G.number_of_nodes())
            degs.append(2.0*G.number_of_edges()/G.number_of_nodes())
        print("Average number of nodes:", np.mean(sizes))
        print("Average degree:", np.mean(degs))

        N = len(graphs)
        sp = list()
        norm = list()
        print("\nGraph preprocessing progress:")
        for g in tqdm(graphs):
            current_sp = dict(nx.all_pairs_dijkstra_path_length(g, cutoff=depth))
            sp.append(current_sp)
            sp_g = nx.Graph()
            for node in current_sp:
                for neighbor in current_sp[node]:
                    if node == neighbor:
                        sp_g.add_edge(node, node, weight=1.0)
                    else:
                        sp_g.add_edge(node, neighbor, weight=1.0/current_sp[node][neighbor])
            M = nx.to_numpy_matrix(sp_g)
            norm.append(np.linalg.norm(M,'fro'))
            
        K = np.zeros((N, N))
        print("\nKernel computation progress:")
        for i in tqdm(range(N)):
            for j in range(i, N):
                K[i,j] = spgk(sp[i], sp[j], norm[i], norm[j])
                K[j,i] = K[i,j]

        self.loader.save(K, 'kernel_matrix')

    def build_data_feature(self, model_name='xlm-roberta-base'):
        '''
        Build sentence features.
        构建句子的特征。
        '''
        if not self.loader.data:
            print("empty data")
            return 

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)

        print("\nSentence of features building progress:")
        encoded_input = tokenizer(self.loader.data[0], return_tensors='pt').to(device)
        feature = model(**encoded_input).pooler_output
        with torch.no_grad():
            for i in tqdm(self.loader.data[1:]):
                encoded_input = tokenizer(i, return_tensors='pt').to(device)
                output = model(**encoded_input).pooler_output
                feature = torch.cat([feature,output], dim=0)
        self.loader.save(feature.to(torch.device("cpu")), 'feature')

    def build_link_list(self, edge_threshold):
        if type(self.loader.kernel_matrix) == 'NoneType':
            print("empty kernel_matrix")
            return 

        print("\nBuild link_list:")
        link_list = np.empty([0,2])
        for i in tqdm(range(self.loader.kernel_matrix.shape[0])):
            for j in range(i+1, self.loader.kernel_matrix.shape[0]):
                if self.loader.kernel_matrix[i,j] >= edge_threshold:
                    edge = np.array([[i,j]])
                    link_list = np.append(link_list,edge,axis=0)

        self.loader.save(link_list, 'link_list')

    def build_wl_embedding(self, max_iter=2):
        if not self.loader.data or type(self.loader.kernel_matrix) == 'NoneType' or type(self.loader.link_list) == 'NoneType':
            print("empty data or kernel_matrix or link_list")
            return 

        node_color_dict = {}
        node_neighbor_dict = {}
        node_list = np.mgrid[:len(self.loader.data)]
        link_list = self.loader.link_list

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

        self.loader.save(node_color_dict, 'wl_embedding')
