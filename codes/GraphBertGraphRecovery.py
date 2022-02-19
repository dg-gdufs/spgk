'''
Description: file description
Version: 1.0
Autor: Renhetian
Date: 2022-02-19 21:54:31
LastEditors: Renhetian
LastEditTime: 2022-02-20 00:29:46
'''

import torch
import torch.optim as optim

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from codes.GraphBert import GraphBert
from codes.Utils import device

import time
import pickle

BertLayerNorm = torch.nn.LayerNorm

class GraphBertGraphRecovery(BertPreTrainedModel):

    save_path = 'data/loader/'
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    output = None

    def __init__(self, gbp, config, lr=0.001, max_epoch=500, weight_decay=5e-4):
        super(GraphBertGraphRecovery, self).__init__(config)
        self.gbp = gbp
        self.config = config
        self.bert = GraphBert(config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epoch = max_epoch
        self.save_path += self.gbp.dataset_name
        self.init_weights()

    def forward(self, raw_features, wl_role_ids, init_pos_ids, hop_dis_ids, idx=None):
        raw_features = raw_features.to(device)
        wl_role_ids = wl_role_ids.to(device)
        init_pos_ids = init_pos_ids.to(device)
        hop_dis_ids = hop_dis_ids.to(device)
        outputs = self.bert(raw_features, wl_role_ids, init_pos_ids, hop_dis_ids)

        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        x_hat = sequence_output
        x_norm = torch.norm(x_hat, p=2, dim=1)
        nume = torch.mm(x_hat, x_hat.t())
        deno = torch.ger(x_norm, x_norm)
        cosine_similarity = nume / deno
        return cosine_similarity

    def train_model(self, max_epoch):
        print('GrapBert, dataset: ' + self.gbp.dataset_name + ', Pre-training, Graph Structure Recovery.')
        t_begin = time.time()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        for epoch in range(max_epoch):
            t_epoch_begin = time.time()
            # -------------------------
            self.train()
            optimizer.zero_grad()

            output = self.forward(self.gbp.graphbert_data['raw_embeddings'], self.gbp.graphbert_data['wl_embedding'], self.gbp.graphbert_data['int_embeddings'], self.gbp.graphbert_data['hop_embeddings'])
            row_num, col_num = output.size()
            loss_train = torch.sum((output - self.gbp.graphbert_data['A'].to(device).to_dense()) ** 2)/(row_num*col_num)

            loss_train.backward()
            optimizer.step()

            self.learning_record_dict[epoch] = {'loss_train': loss_train.item(), 'time': time.time() - t_epoch_begin}
            # -------------------------
            if epoch % 50 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'time: {:.4f}s'.format(time.time() - t_epoch_begin))

        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))
        self.output = output

    def run(self):
        self.train_model(self.max_epoch)
        with open(self.save_path + '/kernel_matrix_graphbert.pkl', 'wb') as f:
            pickle.dump(self.output.detach().numpy(), f)
