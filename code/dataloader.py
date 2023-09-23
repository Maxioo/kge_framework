import torch
import queue

import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import Dataset

def randn(*args):
    return np.random.randn(*args).astype('f')

class TrainDataset(Dataset):
    def __init__(self, train_triples, all_true_triples, headTailSelector, \
                entity2id, id2entity, relation2id, id2relation, neg_ratio=1.0):
        self.train_triples = train_triples
        self.indexes = np.array(list(self.train_triples.keys())).astype(np.int32)
        self.values = np.array(list(self.train_triples.values())).astype(np.float32)

        self.n_words = len(entity2id.keys())
        self.n_relation = len(relation2id.keys())
        self.len = len(self.values)
        self.neg_ratio = int(neg_ratio)
        self.headTailSelector = headTailSelector
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.entity2id = entity2id
        self.id2entity = id2entity

        self.head_dict = all_true_triples['head']
        self.tail_dict = all_true_triples['tail']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.indexes[idx, :]
        negative_samples = np.tile(positive_sample, (self.neg_ratio, 1))
        # negative_samples = np.tile(positive_sample, (self.neg_ratio + self.neg_ratio // 2, 1))

        tmp_index_rel = positive_sample[1]

        pr = self.headTailSelector[tmp_index_rel]
        right_num = int(self.neg_ratio * pr)
        right_exclu = self.tail_dict[(positive_sample[0], positive_sample[1])]
        left_num = self.neg_ratio - right_num
        left_exclu = self.head_dict[(positive_sample[1], positive_sample[2])]

        # sample negative
        tail_negative_list = []
        tail_negative_size = 0

        while tail_negative_size < right_num:
            negative_sample = np.random.randint(self.n_words, size=right_num*2)
            mask = np.in1d(
                negative_sample,
                right_exclu,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            tail_negative_list.append(negative_sample)
            tail_negative_size += negative_sample.size

        if right_num != 0:
            tail_negatives = np.concatenate(tail_negative_list)[:right_num]
            negative_samples[:right_num, 2] = tail_negatives

        head_negative_list = []
        head_negative_size = 0

        while head_negative_size < left_num:
            negative_sample = np.random.randint(self.n_words, size=left_num*2)
            mask = np.in1d(
                negative_sample,
                left_exclu,
                assume_unique=True,
                invert=True
            )
            negative_sample = negative_sample[mask]
            head_negative_list.append(negative_sample)
            head_negative_size += negative_sample.size

#         relation_negative_list = []
#         relation_negative_size = 0
        
        if left_num != 0:
            head_negatives = np.concatenate(head_negative_list)[:left_num]
            negative_samples[right_num:self.neg_ratio, 0] = head_negatives

        # relation_negative_list = []
        # relation_negative_size = 0
        
        # while relation_negative_size < self.neg_ratio // 2:
        #     negative_sample = np.random.randint(self.n_relation, size=self.neg_ratio)
        #     mask = np.in1d(
        #         negative_sample,
        #         [positive_sample[1]],
        #         assume_unique=True,
        #         invert=True
        #     )
        #     negative_sample = negative_sample[mask]
        #     relation_negative_list.append(negative_sample)
        #     relation_negative_size += negative_sample.size

        # relation_negatives = np.concatenate(relation_negative_list)[:self.neg_ratio // 2]
        # negative_samples[self.neg_ratio:, 1] = relation_negatives

        positive_sample = torch.from_numpy(positive_sample)
        negative_sample = torch.from_numpy(negative_samples)
        return positive_sample, negative_sample


    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0).to(torch.long)
        negative_sample = torch.cat([_[1] for _ in data], dim=0).to(torch.long)

        positive_val = torch.full([positive_sample.shape[0]], 1, dtype=torch.long)
        negative_val = torch.full([negative_sample.shape[0]], -1, dtype=torch.long)

        #  samples = torch.cat([positive_sample, negative_sample], dim=0)
        #  values = torch.cat([positive_val, netative_val], dim=0)
        return positive_sample, negative_sample, positive_val, negative_val

class TestDataset(Dataset):
    def __init__(self, valid_triples, all_true_triples, \
                 entity2id, id2entity, relation2id, id2relation, mode, eval_type):
        self.x_valid = np.array(list(valid_triples.keys())).astype(np.int32)
        self.triple_set = set(all_true_triples)
        self.entity_array = np.array(list(id2entity.keys()))
        self.relation2id = relation2id
        self.id2relation = id2relation
        self.entity2id = entity2id
        self.id2entity = id2entity
        self.nentity = len(self.id2entity.keys())
        self.nrelation = len(self.id2relation.keys())
        self.len = len(self.x_valid)
        self.mode = mode
        if self.mode == 'head':
            self.indexdict = all_true_triples['head']
        else:
            self.indexdict = all_true_triples['tail']

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.x_valid[idx]
        new_x_batch = np.tile(self.x_valid[idx], (len(self.entity2id), 1))
        
        if self.mode == 'head':
            new_x_batch[:, 0] = self.entity_array
            tmpTriple = (new_x_batch[0][1], new_x_batch[0][2])
        else:
            new_x_batch[:, 2] = self.entity_array
            tmpTriple = (new_x_batch[0][0], new_x_batch[0][1])
       
        lstIdx = self.indexdict[tmpTriple]
        
        if self.mode == 'head':
            lstIdx.remove(head)
            new_x_batch[lstIdx, 0] = head
        else:
            lstIdx.remove(tail)
            new_x_batch[lstIdx, 2] = tail
            
        filter_bias = np.zeros(self.nentity)
        filter_bias[lstIdx] = -100000
        tmp = torch.LongTensor(new_x_batch)
        filter_bias = torch.FloatTensor(filter_bias)
        negative_sample = tmp

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode

class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
