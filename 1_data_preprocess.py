'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-25 00:35:53
LastEditors: Renhetian
LastEditTime: 2022-02-18 23:40:13
'''

from codes.Preprocess import Preprocess
from codes.DatasetLoader import DatasetLoader

dataset_name = 'subjectivity'
window_size = 2
depth = 1
edge_threshold = 0.2
model_name = 'cahya/bert-base-indonesian-1.5G'
# model_name = 'xlm-roberta-base'


if __name__ == "__main__":
    dl = DatasetLoader(dataset_name)
    pp = Preprocess(dl)
    pp.load_file()
    pp.build_kernel_matrix(window_size, depth)
    pp.build_data_feature(model_name)
    dl.dump(edge_threshold)
