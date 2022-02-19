'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-26 21:02:11
LastEditors: Renhetian
LastEditTime: 2022-02-19 00:49:56
'''

import os
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from codes.Utils import doc_format, spgk, device


class Preprocess:

    load_path = 'data/dataset/'

    def __init__(self, loader) -> None:
        self.loader = loader
        self.load_path += self.loader.dataset_name

    def load_file(self):
        for file_name in os.listdir(self.load_path):
            if file_name.endswith('.neg'):
                with open(self.load_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    for i in file:
                        self.loader.data.append(doc_format(i))
                        self.loader.label.append(0)
            elif file_name.endswith('.pos'):
                with open(self.load_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
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
