'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-25 00:35:53
LastEditors: Renhetian
LastEditTime: 2022-01-26 21:19:08
'''

from Preprocess import Preprocess
from DatasetLoader import DatasetLoader


# python -m script.data_preprocess
if __name__ == "__main__":
    dl = DatasetLoader('subjectivity')
    pp = Preprocess(dl)
    pp.load_file('data/dataset/subjectivity/')
    pp.build_kernel_matrix(window_size=2, depth=1)
    pp.build_data_feature(model_name='cahya/bert-base-indonesian-1.5G')
    # pp.build_data_feature(model_name='xlm-roberta-base')