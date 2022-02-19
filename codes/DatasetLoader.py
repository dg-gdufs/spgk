'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:24:13
LastEditors: Renhetian
LastEditTime: 2022-02-20 00:32:46
'''

import os
import pickle

from tqdm import tqdm


class DatasetLoader:

    save_path = 'data/loader/'
    dump_path = 'data/graphbert_dataset/'
    
    def __init__(self, dataset_name='default') -> None:
        self.dataset_name = dataset_name
        self.save_path += dataset_name
        self.dump_path += dataset_name

        self.data = []
        self.label = []
        self.feature = None
        self.kernel_matrix = None

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)
    
    def save(self, data, save_name):
        save_path = self.save_path + '/{}.pkl'.format(save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, attr='all'):
        if attr == 'data' or attr == 'all':
            try:
                self.data, self.label = pickle.load(open(self.save_path + '/data.pkl', 'rb'))
            except:
                print("data.pkl not found")
        if attr == 'feature' or attr == 'all':
            try:
                self.feature = pickle.load(open(self.save_path + '/feature.pkl', 'rb'))
            except:
                print("feature.pkl not found")
        if attr == 'kernel_matrix' or attr == 'all':
            try:
                if os.path.exists(self.save_path + '/kernel_matrix_graphbert.pkl'):
                    self.kernel_matrix = pickle.load(open(self.save_path + '/kernel_matrix_graphbert.pkl', 'rb'))
                else:
                    self.kernel_matrix = pickle.load(open(self.save_path + '/kernel_matrix.pkl', 'rb'))
            except:
                print("kernel_matrix.pkl not found")

    def dump(self, edge_threshold=0.2):
        if not self.data or type(self.kernel_matrix) == 'NoneType' or type(self.feature) == 'NoneType':
            print("empty data or kernel_matrix or feature")
            return 

        with open(self.dump_path + '/node','w', encoding='utf-8') as file:
            for i in tqdm(range(len(self.data))):
                file.write(str(i))
                file.write('\t')
                for j in self.feature[i]:
                    if j > 0:
                        file.write('1')
                    else:
                        file.write('0')
                    file.write('\t')
                file.write(str(self.label[i]))
                file.write('\n')

        with open(self.dump_path + '/link','w', encoding='utf-8') as file:
            for i in tqdm(range(self.kernel_matrix.shape[0])):
                for j in range(i+1, self.kernel_matrix.shape[0]):
                    if self.kernel_matrix[i,j] >= edge_threshold:
                        file.write(str(i) + '\t' + str(j) + '\n')
