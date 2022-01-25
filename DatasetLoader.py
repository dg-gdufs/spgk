'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-24 18:24:13
LastEditors: Renhetian
LastEditTime: 2022-01-25 14:02:53
'''

import os
import pickle
from Utils import build_data_feature, build_kernel_matrix, doc_format


class DatasetLoader:
    
    def __init__(self, save_path='data/loader/default', dataset_name='default') -> None:
        self.dataset_name = dataset_name
        self.save_path = save_path
        self.data = []
        self.label = []
        self.feature = None
        self.kernel_matrix = None
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def preprocessing(self, window_size=2, depth=1, model_name='xlm-roberta-base'):
        build_kernel_matrix(self, window_size, depth)
        build_data_feature(self, model_name)
    
    def save(self, data, save_name):
        save_path = self.save_path + '/{}-{}.pkl'.format(self.dataset_name, save_name)
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        pass


class FileDatasetLoader(DatasetLoader):

    def __init__(self, save_path='data/loader/default', dataset_name='default') -> None:
        super().__init__(save_path, dataset_name)

    def load(self, file_path):
        for file_name in os.listdir(file_path):
            if file_name.endswith('.neg'):
                with open(file_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    for i in file:
                        self.data.append(doc_format(i))
                        self.label.append(0)
            elif file_name.endswith('.pos'):
                with open(file_path+'/'+file_name, 'r', encoding='utf-8', errors='ignore') as file:
                    for i in file:
                        self.data.append(doc_format(i))
                        self.label.append(1)
        self.save([self.data, self.label], 'data')


class PickleDatasetLoader(DatasetLoader):

    def __init__(self, save_path='data/loader/default', dataset_name='default') -> None:
        super().__init__(save_path, dataset_name)

    def load(self):
        pass