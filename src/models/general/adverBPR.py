# -*- coding: UTF-8 -*-

import torch.nn as nn
import torch

from models.AdverModel import AdverModel
import numpy as np

class adverBPR(AdverModel):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return AdverModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, u_ids, i_ids):
        self.check_list = []
        #u_ids = feed_dict['user_id']  # [batch_size]
        #i_ids = feed_dict['item_id']  # [batch_size, -1]
        
        #u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        cf_u_vectors = self.u_embeddings(u_ids)
        cf_u_vectors = self.apply_filter(cf_u_vectors)
        out_u_vectors = cf_u_vectors
        cf_u_vectors = cf_u_vectors.unsqueeze(1).repeat((1, i_ids.shape[1], 1))

        cf_i_vectors = self.i_embeddings(i_ids)
        # print('cf_u_vectors', cf_u_vectors.shape)
        # print('cf_i_vectors', cf_i_vectors.shape)
        
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
        #prediction = (cf_u_vectors[:, None, :] * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
        #print(prediction.view(len(u_ids), -1).shape)
        #print(out_u_vectors.shape)
            
        return prediction.view(len(u_ids), -1), out_u_vectors

    def model_(self, user, items):
        # user = torch.from_numpy(np.array(user))
        # items = torch.from_numpy(np.array(items))
        user = user.repeat((1, items.shape[0])).squeeze(0)

        cf_u_vectors = self.u_embeddings(user)
        cf_u_vectors = self.apply_filter(cf_u_vectors)
        # cf_u_vectors = cf_u_vectors.repeat((items.shape[0], 1))
        cf_i_vectors = self.i_embeddings(items)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
            
        return prediction


