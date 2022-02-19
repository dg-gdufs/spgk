'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 00:02:37
LastEditors: Renhetian
LastEditTime: 2022-02-19 01:17:10
'''

from codes.GraphBert import GraphBert

dataset_name = 'subjectivity'
max_iter = 2
c = 0.15


if __name__ == "__main__":
    gb = GraphBert(dataset_name)
    gb.build_graphbert_data(c)
    gb.build_wl(max_iter)
