'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 00:02:37
LastEditors: Renhetian
LastEditTime: 2022-02-19 22:06:18
'''

from codes.GraphBertPreprocess import GraphBertPreprocess

dataset_name = 'subjectivity'
max_iter = 2
c = 0.15
k = 5


if __name__ == "__main__":
    gb = GraphBertPreprocess(dataset_name)
    gb.build_graphbert_data(c)
    gb.build_wl(max_iter)
    gb.build_batch(k)
    gb.build_hop(k)
