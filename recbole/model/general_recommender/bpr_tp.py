# -*- coding: utf-8 -*-
# @Time   : 2020/6/25
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/9/16
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
BPR
################################################
Reference:
    Steffen Rendle et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." in UAI 2009.
"""

import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss_TP_inb,BPRLoss_hard_inb,BPRLoss_OP_DRO_Triplet,BPRLoss_inb,BPRLoss_OP_DRO_Triplet_Auto
from recbole.utils import InputType
import numpy as np

class BPR_TP(GeneralRecommender):
    r"""BPR is a basic matrix factorization model that be trained in the pairwise way."""
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(BPR_TP, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]

        # define layers and loss
        self.config=config
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        # self.loss = BPRLoss_TP_inb(base_loss=config['base_loss'],gamma=config['gamma'],phi_class=config['phi_class'])
        # self.loss=BPRLoss_hard_inb(topM=500)
        # self.loss=BPRLoss_OP_DRO_Triplet(lam1=1.5,lam2=30,beta=1e-5)
        self.loss=BPRLoss_OP_DRO_Triplet_Auto(config=config)
        # self.loss=BPRLoss_inb(beta=1e-5)
        self.count_idx=0
        self.epoch_idx=0
        # print(config['base_loss'],config['gamma'],config['phi_class'])
        # parameters initialization
        self.apply(xavier_normal_initialization)
    def neg_sample(self,bs,neg_num):
        neg_idx=np.random.choice(self.n_items,[bs,neg_num])
        return torch.Tensor(neg_idx).long()
    def get_user_embedding(self, user):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction,weight_dict=None):
        # self.count_idx+=1
        # self.loss.gamma=max(1.0,5*(0.999)**self.count_idx)
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID]
        neg_item=self.neg_sample(len(user),self.config['neg_sample']).to(user.device)
        # print(neg_item)
        # print(pos_item.shape,user.shape,neg_item.shape)
        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        weight_dict['user']=user
        weight_dict['item']=pos_item
        # if weight_dict:
        #     weight_list=[]
        #     for k in range(len(user)):
        #         u,i=user[k].cpu().item(),pos_item[k].cpu().item()
        #         weight_list.append(weight_dict[u][i])
        #     weight_list=torch.Tensor(weight_list).to(user.device)
        # pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
        #     user_e, neg_e
        # ).sum(dim=1)
        if weight_dict:
            loss = self.loss(user_e, pos_e,neg_e,epoch_idx=self.epoch_idx,weight_dict=weight_dict)
        else:
            loss = self.loss(user_e, pos_e, neg_e, epoch_idx=self.epoch_idx, weight_dict=None)
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)
