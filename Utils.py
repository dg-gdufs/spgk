'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:54:30
LastEditors: Renhetian
LastEditTime: 2022-01-26 21:19:44
'''

import re
import torch

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
