import logging
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from datetime import datetime

from dataloader import TestDataset

class TransE(nn.Module):
    def __init__(self, model_name, nentity, nrelation, args):
        super(TransE, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.lmbda = args.lmbda
        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]), 
            requires_grad=False
        )

        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim

        self.entity_embedding = nn.Embedding(nentity, self.entity_dim)
        self.relation_embedding = nn.Embedding(nentity, self.relation_dim)

        self.init_parameters()

    def init_parameters(self):
        if not self.args.use_init:
            nn.init.xavier_uniform_(self.entity_embedding.weight.data)
            nn.init.xavier_uniform_(self.relation_embedding.weight.data)

    def load_embedding(self, init_ent_embs, init_rel_embs):

        init_ent_embs = torch.from_numpy(init_ent_embs)
        init_rel_embs = torch.from_numpy(init_rel_embs)

        if self.args.cuda:
            init_ent_embs = init_ent_embs.cuda()
            init_rel_embs = init_rel_embs.cuda()

        self.entity_embedding.weight.data = init_ent_embs
        self.relation_embedding.weight.data = init_rel_embs
        
        print("Load form Embedding success !")

    def loss(self, positive_score, negative_score):
        if self.args.negative_adversarial_sampling:
            negative_score = (F.softmax(negative_score * self.args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = F.logsigmoid(positive_score)

        positive_sample_loss = - positive_score.mean()
        negative_sample_loss = - negative_score.mean()
        loss = (positive_sample_loss + negative_sample_loss)/2
        if self.args.regularization != 0.0:
            #Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                self.entity_embedding.norm(p = 3)**3 + 
                self.relation_embedding.norm(p = 3).norm(p = 3)**3
            )
            loss = loss + regularization
        return loss

    def forward(self, px, nx, py, ny):

        ph = self.entity_embedding(px[:,0])
        pr = self.relation_embedding(px[:,1])
        pt = self.entity_embedding(px[:,2])
        positive_score = self._calc(ph, pr, pt)

        nh = self.entity_embedding(nx[:,0])
        nr = self.relation_embedding(nx[:,1])
        nt = self.entity_embedding(nx[:,2])
        negative_score = self._calc(nh, nr, nt).reshape([-1, self.args.neg_ratio])

        return self.loss(positive_score, negative_score)

    def _calc(self, h, r, t):
        score = h + r - t
        score = self.gamma.item() - torch.norm(score, p=1, dim=1)
        return score.squeeze()

    def predict(self, x):

        h = self.entity_embedding(x[:, 0])
        r = self.relation_embedding(x[:, 1])
        t = self.entity_embedding(x[:, 2])

        score = self._calc(h, r, t)

        return score
    

def train_step(model, optimizer, train_iterator, args):
    '''
    A single train step. Apply back-propation and return the loss
    '''

    model.train()
    optimizer.zero_grad()

    positive_batch, negative_batch, yp_batch, yn_batch = next(train_iterator)

    if args.cuda:
        positive_batch = positive_batch.cuda()
        negative_batch = negative_batch.cuda()
        yp_batch = yp_batch.cuda()
        yn_batch = yn_batch.cuda()

    loss = model(positive_batch, negative_batch, yp_batch, yn_batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()

    log = {
        'loss': loss.item()
    }

    return log

def test_step(model, valid_triples, test_dataset_list, entity2id, id2entity, relation2id, id2relation, relation2type, args):
    '''
    Evaluate the model on test or valid datasets
    '''
    model.eval()
    #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
    #Prepare dataloader for evaluation

    logs = []
    detail_logs = [[[],[],[],[]], [[],[],[],[]]]
    dd = ["N-N", "N-1", "1-N", "1-1"]
    mm = {'head':0, 'tail':1}

    step = 0
    total_steps = sum([len(dataset) for dataset in test_dataset_list])

    with torch.no_grad():
        for test_dataset in test_dataset_list:
            for positive_sample, negative_sample, filter_bias, mode in test_dataset:

                if args.cuda:
                    s = "cuda:0"
                    positive_sample = positive_sample.to(s, non_blocking=True)
                    negative_sample = negative_sample.to(s, non_blocking=True)
                    filter_bias = filter_bias.to(s, non_blocking=True)

                batch_size = positive_sample.size(0)

                nentity = negative_sample.size(1)
                score = model.predict(negative_sample.reshape([-1, 3]))
                score = score.reshape([-1, nentity])
                score += filter_bias
                
                argsort = torch.argsort(score, dim = 1, descending=True)
                
                if mode == 'head':
                    positive_arg = positive_sample[:, 0]
                elif mode == 'tail':
                    positive_arg = positive_sample[:, 2]
                else:
                    raise ValueError('mode %s not supported' % mode)

                check_ids = []
                for i in range(batch_size):
                    relation = positive_sample[i][1]
                    #Notice that argsort is not ranking
                    ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                    assert ranking.size(0) == 1

                    #ranking + 1 is the true ranking used in evaluation metrics
                    ranking = 1 + ranking.item()
                    logs.append({
                        'MRR': 1.0/ranking,
                        'MR': float(ranking),
                        'HITS@1': 1.0 if ranking <= 1 else 0.0,
                        'HITS@3': 1.0 if ranking <= 3 else 0.0,
                        'HITS@10': 1.0 if ranking <= 10 else 0.0,
                    })
                    ttype = relation2type[relation.item()]
                    detail_logs[mm[mode]][ttype].append({
                        'MRR_' + dd[ttype] + '_' + mode: 1.0/ranking,
                        'MR_' + dd[ttype] + '_' + mode: float(ranking),
                        'HITS@1_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 1 else 0.0,
                        'HITS@3_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 3 else 0.0,
                        'HITS@10_' + dd[ttype] + '_' + mode: 1.0 if ranking <= 10 else 0.0,
                    })
                if step % args.test_log_steps == 0:
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

    metrics = {}
    for metric in logs[0].keys():
        metrics[metric] = sum([log[metric] for log in logs])/len(logs)
    for i in range(2):
        for j in range(4):
            for metric in detail_logs[i][j][0].keys():
                metrics[metric] = sum([log[metric] for log in detail_logs[i][j]])/len(detail_logs[i][j])
    return metrics