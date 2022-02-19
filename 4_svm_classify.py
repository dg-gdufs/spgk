'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-01-27 21:11:05
LastEditors: Renhetian
LastEditTime: 2022-02-20 00:56:38
'''

from codes.DatasetLoader import DatasetLoader
from codes.SVMClassification import SVMClassification

cv = 10
test_size = 0.3


if __name__ == "__main__":
    dl = DatasetLoader('subjectivity')
    dl.load()
    svm = SVMClassification(dl, cv=cv, test_size=test_size)
    svm.run()