import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import utils

class Discriminator(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument('--neg_slope', type=float,
                            default=0.2,
                            help='negative slope for leakyReLU.')
        return parser

    def __init__(self, embed_dim, model_path, random_seed=2020, dropout=0.3, neg_slope=0.2,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.embed_dim = int(embed_dim)
        #self.feature_info = feature_info
        self.random_seed = random_seed
        self.dropout = dropout
        self.neg_slope = neg_slope
        self.criterion = nn.NLLLoss()
        # self.out_dim = feature_info.num_class
        self.out_dim = 2
        # self.name = feature_info.name
        # self.model_file_name = \
        #     self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_path = model_path
        self.optimizer = None

        # 7 layers
        # ML-1M
        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
  
    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    # def save_model(self, model_path=None):
    #     if model_path is None:
    #         model_path = self.model_path
    #     dir_path = os.path.dirname(model_path)
    #     if not os.path.exists(dir_path):
    #         os.mkdir(dir_path)
    #     torch.save(self.state_dict(), model_path)
    #     # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    # def load_model(self, model_path=None):
    #     if model_path is None:
    #         model_path = self.model_path
    #     self.load_state_dict(torch.load(model_path))
    #     self.eval()
    #     #logging.info('Load ' + self.name + ' discriminator model from ' + model_path)

    def save_model(self, model_path=None, add_path=None):
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        utils.check_dir(model_path)
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                    model_path)
        #logging.info('Save model to ... ' + model_path[50:])

    def load_model(self, model_path=None, add_path=None, flag=0):
        if model_path is None:
            model_path = self.model_path
        if add_path:
            model_path += add_path
        
        if torch.cuda.is_available():
            check_point = torch.load(model_path)
        else:
            check_point = torch.load(model_path, map_location=torch.device('cpu'))
            
        self.load_state_dict(check_point['model_state_dict'])
        if flag == 0:
            self.optimizer.load_state_dict(check_point['optimizer_state_dict'])
        #logging.info('Load model from ' + model_path)








class BinaryDiscriminator(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument('--neg_slope', type=float,
                            default=0.2,
                            help='negative slope for leakyReLU.')
        return parser

    def __init__(self, embed_dim, feature_info, random_seed=2020, dropout=0.3, neg_slope=0.2,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_info = feature_info
        self.random_seed = random_seed
        self.dropout = dropout
        self.neg_slope = neg_slope
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.out_dim = 1
        self.name = feature_info.name
        self.model_file_name = \
            self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        # For Ml-1M
        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )

        # Insurance
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )

        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     # # nn.BatchNorm1d(num_features=self.embed_dim),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
        # )
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     # # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
        # )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.FloatTensor).cuda()
        else:
            labels = labels.type(torch.FloatTensor)
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = self.sigmoid(scores)

        # prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        if torch.cuda.device_count() > 0:
            threshold = torch.tensor([0.5]).cuda()
        else:
            threshold = torch.tensor([0.5])
        prediction = (output > threshold).float() * 1

        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        #logging.info('Load ' + self.name + ' discriminator model from ' + model_path)


class BinaryAttacker(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument('--neg_slope', type=float,
                            default=0.2,
                            help='negative slope for leakyReLU.')
        return parser

    def __init__(self, embed_dim, feature_info, snap_iter, random_seed=2020, dropout=0.3, neg_slope=0.2,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_info = feature_info
        self.random_seed = random_seed
        self.dropout = dropout
        self.neg_slope = neg_slope
        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.out_dim = 1
        self.name = feature_info.name
        self.model_file_name = \
            self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_file_name += '_snap{}'.format(snap_iter)
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        # For Insurance
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )

        # For Ml-1M
        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            # nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.3),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        )


        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     # # nn.BatchNorm1d(num_features=self.embed_dim),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
        # )
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 4), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     # # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(p=0.3),
        #     # nn.Linear(int(self.embed_dim), int(self.embed_dim), bias=True),
        #     # nn.LeakyReLU(0.2),
        #     # nn.Dropout(p=0.3),
        #     nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
        # )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        if torch.cuda.device_count() > 0:
            labels = labels.cpu().type(torch.FloatTensor).cuda()
        else:
            labels = labels.type(torch.FloatTensor)
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = self.sigmoid(scores)

        # prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        if torch.cuda.device_count() > 0:
            threshold = torch.tensor([0.5]).cuda()
        else:
            threshold = torch.tensor([0.5])
        prediction = (output > threshold).float() * 1

        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        #logging.info('Load ' + self.name + ' discriminator model from ' + model_path)


class MultiClassAttacker(nn.Module):
    @staticmethod
    def parse_disc_args(parser):
        parser.add_argument('--neg_slope', type=float,
                            default=0.2,
                            help='negative slope for leakyReLU.')
        return parser

    def __init__(self, embed_dim, feature_info, snap_iter, random_seed=2020, dropout=0.3, neg_slope=0.2,
                 model_dir_path='../model/Model/', model_name=''):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.feature_info = feature_info
        self.random_seed = random_seed
        self.dropout = dropout
        self.neg_slope = neg_slope
        self.criterion = nn.NLLLoss()
        self.out_dim = feature_info.num_class
        self.name = feature_info.name
        self.model_file_name = \
            self.name + '_' + model_name + '_disc.pt' if model_name != '' else self.name + '_disc.pt'
        self.model_file_name += '_snap{}'.format(snap_iter)
        self.model_path = os.path.join(model_dir_path, self.model_file_name)
        self.optimizer = None

        # 7 layers
        self.network = nn.Sequential(
            nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
            nn.BatchNorm1d(num_features=self.embed_dim * 2),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
            nn.BatchNorm1d(num_features=self.embed_dim * 4),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
            nn.BatchNorm1d(num_features=self.embed_dim),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim), int(self.embed_dim / 2), bias=True),
            nn.LeakyReLU(self.neg_slope),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(self.embed_dim / 2), self.out_dim, bias=True)
            )
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 4), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 2),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim * 4), int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
        # )
        # self.network = nn.Sequential(
        #     nn.Linear(self.embed_dim, int(self.embed_dim * 2), bias=True),
        #     # nn.BatchNorm1d(num_features=self.embed_dim * 4),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim * 2), bias=True),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim * 2), int(self.embed_dim), bias=True),
        #     nn.LeakyReLU(self.neg_slope),
        #     nn.Dropout(p=self.dropout),
        #     nn.Linear(int(self.embed_dim), self.out_dim, bias=True)
        # )

    @staticmethod
    def init_weights(m):
        """
        initialize nn weights，called in main.py
        :param m: parameter or the nn
        :return:
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                torch.nn.init.normal_(m.bias, mean=0.0, std=0.01)
        elif type(m) == torch.nn.Embedding:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, embeddings, labels):
        output = self.predict(embeddings)['output']
        loss = self.criterion(output.squeeze(), labels)
        return loss

    def predict(self, embeddings):
        scores = self.network(embeddings)
        output = F.log_softmax(scores, dim=1)
        prediction = output.max(1, keepdim=True)[1]     # get the index of the max
        result_dict = {'output': output,
                       'prediction': prediction}
        return result_dict

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        dir_path = os.path.dirname(model_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        torch.save(self.state_dict(), model_path)
        # logging.info('Save ' + self.name + ' discriminator model to ' + model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.load_state_dict(torch.load(model_path))
        self.eval()
        logging.info('Load ' + self.name + ' discriminator model from ' + model_path)
