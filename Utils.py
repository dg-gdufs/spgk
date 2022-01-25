'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:54:30
LastEditors: Renhetian
LastEditTime: 2022-01-25 12:30:50
'''

import re
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def doc_format(string):
    '''
    Tokenization/string cleaning for all datasets.
    清理数据内包含的标记/字符串。
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


def build_kernel_matrix(loader, window_size, depth):
    '''
    Build kernel matrices.
    构建图核。
    '''
    graphs = list()
    sizes = list()
    degs = list()
    print("\nGraphs of words building progress:")
    for doc in tqdm(loader.data):
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

    loader.kernel_matrix = K
    loader.save(loader.kernel_matrix, 'kernel_matrix')


def spgk(sp_g1, sp_g2, norm1, norm2):
    '''
    Compute spgk kernel.
    计算spgk内核。
    '''
    if norm1 == 0 or norm2 == 0:
        return 0
    else:
        kernel_value = 0
        for node1 in sp_g1:
            if node1 in sp_g2:
                kernel_value += 1
                for node2 in sp_g1[node1]:
                    if node2 != node1 and node2 in sp_g2[node1]:
                        kernel_value += (1.0/sp_g1[node1][node2]) * (1.0/sp_g2[node1][node2])

        kernel_value /= (norm1 * norm2)

        return kernel_value


def build_data_feature(loader, model_name):
    '''
    Build sentence features.
    构建句子的特征。
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    print("\nSentence of features building progress:")
    encoded_input = tokenizer(loader.data[0], return_tensors='pt')
    loader.feature = model(**encoded_input).pooler_output
    for i in tqdm(loader.data[1:]):
        encoded_input = tokenizer(i, return_tensors='pt')
        output = model(**encoded_input).pooler_output
        loader.feature = torch.cat([loader.feature,output], dim=0)
    loader.save(loader.feature, 'feature')
