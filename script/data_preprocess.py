'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-25 00:35:53
LastEditors: Renhetian
LastEditTime: 2022-01-25 12:40:21
'''

from DatasetLoader import FileDatasetLoader


if __name__ == "__main__":
    fdl = FileDatasetLoader('subjectivity')
    fdl.load('data/dataset/subjectivity/')
    fdl.preprocessing(window_size=2, depth=1, model_name='cahya/bert-base-indonesian-1.5G')
    # fdl.preprocessing()