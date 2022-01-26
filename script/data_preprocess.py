'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-25 00:35:53
LastEditors: Renhetian
LastEditTime: 2022-01-27 02:07:40
'''

from Preprocess import Preprocess
from DatasetLoader import DatasetLoader

dataset_name = 'subjectivity'
window_size = 2
depth = 1
model_name = 'cahya/bert-base-indonesian-1.5G'
# model_name = 'xlm-roberta-base'


# python -m script.data_preprocess
if __name__ == "__main__":
    dl = DatasetLoader(dataset_name)
    dl.load()
    pp = Preprocess(dl)
    pp.load_file('data/dataset/' + dataset_name)
    pp.build_kernel_matrix(window_size, depth)
    pp.build_data_feature(model_name)
    # pp.build_wl_embedding()