# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class BPRLoss_OP_DRO_Triplet_Auto(nn.Module):
    def __init__(self, config,beta=1e-10,lam1=1.0,lam2=1.0):
        super(BPRLoss_OP_DRO_Triplet_Auto, self).__init__()
        self.neg_weight=config['neg_weight']
        self.neg_margin=config['neg_margin']
        self.beta = config['beta']
        self.alpha = config['alpha']
        self.lam1=config['lam1']
        self.lam2=config['lam2']
        self.inner=config['inner']
        self.type=config['type']
        self.drop_rate=config['drop_rate']
        self.exponent=config['exponent']
        self.num_gradual=config['num_gradual']
        print('loss_type:',self.type)
    def drop_rate_schedule(self,iteration):

        drop_rate = np.linspace(0, self.drop_rate ** self.exponent, self.num_gradual)
        if iteration < self.num_gradual:
            return drop_rate[iteration]
        else:
            return self.drop_rate
    def softplus(self,x,k=5):
        return torch.log(1+torch.exp(k*x))/k
    def forward(self,uemb,pemb,nemb,epoch_idx,weight_dict=None,a=None,b=None,gamma=None,theta_a=None,theta_b=None,sp=None,sn=None):
        if self.type not in ['CCL']:
            neg_score=(uemb.unsqueeze(1)*nemb).sum(-1)
            pos_score = (uemb * pemb).sum(1, keepdim=True)

        if self.type == 'TP_Point_TP':
            a=torch.clip(a,0,1)
            b=torch.clip(b,0,1)
            gamma=torch.clip(gamma,-1,1)
            theta_a=torch.clip(theta_a,0,1e9)
            theta_b=torch.clip(theta_b,0,1e9)
            sp = torch.clip(sp, -1, 4)
            sn=torch.clip(sn,0,5)
            pos_score=torch.sigmoid(pos_score)
            neg_score=torch.sigmoid(neg_score)
            max_val_p = torch.log(1+torch.exp(5*(-torch.square(pos_score - a) + \
                          2 * (1 + gamma) * pos_score - sp)))/5
            max_val_n = torch.log(1+torch.exp(5*(torch.square(neg_score - b) + \
                          2 * (1 + gamma) * neg_score - sn)))/5
            loss = -sp - torch.mean(max_val_p)/self.alpha + \
                  sn + torch.mean(max_val_n)/self.beta + \
                   -gamma**2- theta_b * (b-1-gamma) + theta_a * (a+gamma)
        elif self.type == 'TP_Point_OP':
            a=torch.clip(a,0,1)
            b=torch.clip(b,0,1)
            gamma=torch.clip(gamma,-1,1)
            theta_b=torch.clip(theta_b,0,1e9)
            sn=torch.clip(sn,0,5)
            pos_score=torch.sigmoid(pos_score)
            neg_score=torch.sigmoid(neg_score)

            max_val_n = torch.log(1+torch.exp(5*(torch.square(neg_score - b) + \
                          2 * (1 + gamma) * neg_score - sn)))/5
            loss = torch.mean(torch.square(pos_score - a) - \
                          2 * (1 + gamma) * pos_score) + (self.beta * sn + \
                    torch.mean(max_val_n))/self.beta - theta_b * (b-1-gamma)+0*(sp)

        elif self.type == 'BPR':
            pred=pos_score-neg_score
            loss=-torch.log(torch.sigmoid(pred)+1e-5).sum(1).mean()
        elif self.type=='BCE':

            neg_score=neg_score.reshape(-1)
            pos_score=pos_score.reshape(-1)
            drop_rate=self.drop_rate_schedule(epoch_idx)
            y=torch.cat([pos_score,neg_score])
            t=torch.cat([torch.ones(len(pos_score)),torch.zeros(len(neg_score))]).to(pos_score.device)
            loss = F.binary_cross_entropy_with_logits(y, t)

        elif self.type=='CCL':
            uemb=F.normalize(uemb,dim=-1)
            pemb=F.normalize(pemb,dim=-1)
            nemb=F.normalize(nemb,dim=-1)
            neg_score = (uemb.unsqueeze(1) * nemb).sum(-1)
            pos_score = (uemb * pemb).sum(1)
            pos_loss=torch.relu(1-pos_score)
            neg_loss=self.neg_weight*torch.relu(neg_score-self.neg_margin).mean(-1)
            loss=(pos_loss+neg_loss).mean()

        elif self.type=='softmax':
            pred=torch.exp(neg_score-pos_score)
            loss=torch.log(1+pred.sum(1)).mean()
        else:
            loss=None
        return loss

class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss
