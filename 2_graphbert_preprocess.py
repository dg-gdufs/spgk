'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 00:02:37
LastEditors: Renhetian
LastEditTime: 2022-02-19 22:25:20
'''

from codes.GraphBertPreprocess import GraphBertPreprocess

dataset_name = 'subjectivity'
max_iter = 2
c = 0.15
k = 5


if __name__ == "__main__":
    gbp = GraphBertPreprocess(dataset_name)
    gbp.build_graphbert_data(c)
    gbp.build_wl(max_iter)
    gbp.build_batch(k)
    gbp.build_hop(k)
