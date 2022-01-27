'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-27 21:11:05
LastEditors: Renhetian
LastEditTime: 2022-01-27 21:21:47
'''

from codes.DatasetLoader import DatasetLoader
from codes.SVMClassification import SVMClassification


if __name__ == "__main__":
    dl = DatasetLoader('subjectivity')
    dl.load()
    svm = SVMClassification(dl)
    svm.run()