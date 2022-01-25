'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-25 00:35:53
LastEditors: Renhetian
LastEditTime: 2022-01-25 13:41:54
'''

from DatasetLoader import FileDatasetLoader


if __name__ == "__main__":
    fdl = FileDatasetLoader(save_path='data/loader/subjectivity', dataset_name='subjectivity')
    fdl.load(file_path='data/dataset/subjectivity/')
    fdl.preprocessing(window_size=2, depth=1, model_name='cahya/bert-base-indonesian-1.5G')
    # fdl.preprocessing()