'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 21:54:02
LastEditors: Renhetian
LastEditTime: 2022-02-19 23:14:08
'''

from codes.GraphBertGraphRecovery import GraphBertGraphRecovery
from codes.BertComp import GraphBertConfig
from codes.GraphBertPreprocess import GraphBertPreprocess
from codes.Utils import device

dataset_name = 'subjectivity'
weight_decay = 5e-4
max_epoch = 200
max_iter = 2
lr = 0.01
k = 5

bc = GraphBertConfig(
    residual_type = 'graph_raw',
    x_size = 768, # 节点feature
    y_size = 2, # 几种label类型
    k = k,
    max_wl_role_index = 100,
    max_hop_dis_index = 100,
    max_inti_pos_index = 100,
    hidden_size = 32,
    num_hidden_layers = 2,
    num_attention_heads = 2,
    intermediate_size = 32,
    hidden_act = "gelu",
    hidden_dropout_prob = 0.5,
    attention_probs_dropout_prob = 0.3,
    initializer_range = 0.02,
    layer_norm_eps = 1e-12,
    is_decoder = False,
)


if __name__ == "__main__":
    gbp = GraphBertPreprocess(dataset_name)
    gbp.load(max_iter=max_iter, k=k)
    gbgr = GraphBertGraphRecovery(gbp=gbp, config=bc, lr=lr, max_epoch=max_epoch, weight_decay=weight_decay).to(device)
    gbgr.run()
    
