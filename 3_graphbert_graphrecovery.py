'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 21:54:02
LastEditors: Renhetian
LastEditTime: 2022-02-19 22:06:09
'''

from codes.GraphBertPreprocess import GraphBertPreprocess

dataset_name = 'subjectivity'
max_iter = 2
k = 5


if __name__ == "__main__":
    gb = GraphBertPreprocess(dataset_name)
    gb.load(max_iter=max_iter, k=k)
    
