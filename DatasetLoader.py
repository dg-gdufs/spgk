'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:24:13
LastEditors: Renhetian
LastEditTime: 2022-01-27 02:16:16
'''

import os
import pickle

from tqdm import tqdm


class DatasetLoader:

    save_path = 'data/loader/'
    
    def __init__(self, dataset_name='default') -> None:
        self.dataset_name = dataset_name
        self.save_path += dataset_name
        self.data = []
        self.label = []
        self.feature = None
        self.kernel_matrix = None
        self.edge_threshold = 0.2
        self.wl_embedding = None
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def save(self, data, save_name):
        save_path = self.save_path + '/{}.pkl'.format(save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, attr='all'):
        if attr == 'data' or attr == 'all':
            self.data, self.label = pickle.load(open(self.save_path + '/data.pkl', 'rb'))
        if attr == 'feature' or attr == 'all':
            self.feature = pickle.load(open(self.save_path + '/feature.pkl', 'rb'))
        if attr == 'kernel_matrix' or attr == 'all':
            self.kernel_matrix = pickle.load(open(self.save_path + '/kernel_matrix.pkl', 'rb'))

    def dump(self):
        if not self.data or type(self.kernel_matrix) == 'NoneType' or type(self.feature) == 'NoneType':
            print("empty data or kernel_matrix or feature")
            return 

        with open(self.save_path + '/node','w', encoding='utf-8') as file:
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

        with open(self.save_path + '/link','w', encoding='utf-8') as file:
            for i in tqdm(range(self.kernel_matrix.shape[0])):
                for j in range(i+1, self.kernel_matrix.shape[0]):
                    if self.kernel_matrix[i,j] >= self.edge_threshold:
                        file.write(str(i) + '\t' + str(j) + '\n')