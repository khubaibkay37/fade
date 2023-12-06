# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch

from models.MetaModel import MetaModel
import numpy as np

class BPR(MetaModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return MetaModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)
        # self.u_bias = nn.Embedding(self.user_num, 1)
        # self.i_bias = nn.Embedding(self.item_num, 1)

    def forward(self, u_ids, i_ids, flag):
        self.check_list = []
        #u_ids = feed_dict['user_id']  # [batch_size]
        #i_ids = feed_dict['item_id']  # [batch_size, -1]
        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)
        # user_bias = self.u_bias(u_ids).squeeze(-1)
        # item_bias = self.i_bias(i_ids).squeeze(-1)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
        #prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]

        # if 'attempt0' in flag:
        #     prediction = prediction
        # else:
        #prediction = prediction + user_bias + item_bias
            
        return prediction.view(len(u_ids), -1)

    def model_(self, user, items, flag):
        # user = torch.from_numpy(np.array(user))
        # items = torch.from_numpy(np.array(items))
        user = user.repeat((1, items.shape[0])).squeeze(0)

        cf_u_vectors = self.u_embeddings(user)
        cf_i_vectors = self.i_embeddings(items)
        # user_bias = self.u_bias(user).squeeze(-1)
        # item_bias = self.i_bias(items).squeeze(-1)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
        # if 'attempt0' in flag:
        #     prediction = prediction
        #     print('ORI BPR')
        # else:
        #prediction = prediction + user_bias + item_bias
            
        return prediction


