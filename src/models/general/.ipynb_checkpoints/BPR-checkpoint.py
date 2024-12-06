# -*- coding: UTF-8 -*-
import torch.nn as nn
from models.Model import Model
import torch

class BPR(Model):
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        return Model.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.user_num = corpus.n_users
        super().__init__(args, corpus)

    def _define_params(self):
        self.u_embeddings = nn.Embedding(self.user_num, self.emb_size)
        self.i_embeddings = nn.Embedding(self.item_num, self.emb_size)

    def forward(self, u_ids, i_ids, flag):
        self.check_list = []
        # print(type(i_ids))
        # print(min(u_ids), max(u_ids), torch.min(i_ids), torch.max(i_ids), self.user_num, self.emb_size, self.item_num)
        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        
        cf_u_vectors = self.u_embeddings(u_ids)
        cf_i_vectors = self.i_embeddings(i_ids)

        # print("\n\nIn BPRRR: " ,cf_u_vectors.shape, (cf_i_vectors.shape))
        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)  # [batch_size, -1]
            
        return prediction.view(len(u_ids), -1)

    def model_(self, user, items, flag):
        user = user.repeat((1, items.shape[0])).squeeze(0)

        cf_u_vectors = self.u_embeddings(user)
        cf_i_vectors = self.i_embeddings(items)

        prediction = (cf_u_vectors * cf_i_vectors).sum(dim=-1)
            
        return prediction


