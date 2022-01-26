'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:24:13
LastEditors: Renhetian
LastEditTime: 2022-01-27 00:31:16
'''

import os
import pickle


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
