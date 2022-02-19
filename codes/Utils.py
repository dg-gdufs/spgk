'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:54:30
LastEditors: Renhetian
LastEditTime: 2022-02-19 19:19:55
'''

import re
import time
import torch
import numpy as np
import scipy.sparse as sp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def adj_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
    return labels_onehot


def time_now_formate():
    '''
    返回当前的格式化时间，格式：%Y-%m-%d %H:%M:%S
    '''
    return time.strftime("%Y-%m-%d %H;%M;%S", time.localtime(time.time()))
