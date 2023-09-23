import datetime
import argparse
import logging
import json
import torch
import os

import numpy as np

def override_config(args):
    '''
    Override model and data configuration
    '''
    
    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)
    
    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']

def save_model(model, optimizer, save_variable_list, args, is_best_model=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    save_path = "%s/best/"%args.save_path if is_best_model else  args.save_path
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )
    
#     entity_embedding = model.entity_embedding.detach().cpu().numpy()
#     np.save(
#         os.path.join(args.save_path, 'entity_embedding'), 
#         entity_embedding
#     )
    
#     relation_embedding = model.relation_embedding.detach().cpu().numpy()
#     np.save(
#         os.path.join(args.save_path, 'relation_embedding'), 
#         relation_embedding
#     )

def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    now_time = str(datetime.datetime.now()).replace(" ", ":")
    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, now_time + 'train' + args.model + '.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, now_time + 'test' + args.model + '.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))
        
def parse_line(line):
    line = line.strip().split()
    sub = line[0]
    rel = line[1]
    obj = line[2]
    val = [1]
    if len(line) > 3:
        if line[3] == '-1':
            val = [-1]
    return sub, obj, rel, val


def load_triples_from_txt(filename, parse_line, entity2id, relation2id):

    data = dict()

    with open(filename) as f:
        lines = f.readlines()

    for _, line in enumerate(lines):
        sub, obj, rel, val = parse_line(line)

        sub_ind = entity2id[sub]
        rel_ind = relation2id[rel]
        obj_ind = entity2id[obj]

        data[(sub_ind, rel_ind, obj_ind)] = val

    return data

def read_from_id(filename='../data/WN18RR/entity2id.txt'):
    entity2id = {}
    id2entity = {}
    with open(filename) as f:
        for line in f:
            if len(line.strip().split()) > 1:
                tmp = line.strip().split()
                entity2id[tmp[0]] = int(tmp[1])
                id2entity[int(tmp[1])] = tmp[0]
    return entity2id, id2entity

def build_data(path):
    folder = path

    entity2id, id2entity = read_from_id(folder + '/entity2id.txt')
    relation2id, id2relation = read_from_id(folder + '/relation2id.txt')

    train_triples = load_triples_from_txt(filename=os.path.join(folder, 'train.txt'), parse_line=parse_line, entity2id=entity2id, relation2id=relation2id)

    valid_triples = load_triples_from_txt(filename=os.path.join(folder, 'valid.txt'), parse_line=parse_line, entity2id=entity2id, relation2id=relation2id)

    test_triples = load_triples_from_txt(filename=os.path.join(folder, 'test.txt'), parse_line=parse_line, entity2id=entity2id, relation2id=relation2id)


    left_entity = {}
    right_entity = {}

    with open(os.path.join(folder, 'train.txt')) as f:
        lines = f.readlines()
    for _, line in enumerate(lines):
        head, tail, rel, val = parse_line(line)
        # count the number of occurrences for each (heal, rel)
        if relation2id[rel] not in left_entity:
            left_entity[relation2id[rel]] = {}
        if entity2id[head] not in left_entity[relation2id[rel]]:
            left_entity[relation2id[rel]][entity2id[head]] = 0
        left_entity[relation2id[rel]][entity2id[head]] += 1
        # count the number of occurrences for each (rel, tail)
        if relation2id[rel] not in right_entity:
            right_entity[relation2id[rel]] = {}
        if entity2id[tail] not in right_entity[relation2id[rel]]:
            right_entity[relation2id[rel]][entity2id[tail]] = 0
        right_entity[relation2id[rel]][entity2id[tail]] += 1

    left_avg = {}
    for i in range(len(relation2id)):
        left_avg[i] = sum(left_entity[i].values()) * 1.0 / len(left_entity[i])

    right_avg = {}
    for i in range(len(relation2id)):
        right_avg[i] = sum(right_entity[i].values()) * 1.0 / len(right_entity[i])

    headTailSelector = {}
    for i in range(len(relation2id)):
        headTailSelector[i] = right_avg[i] / (right_avg[i] + left_avg[i])

    return train_triples, valid_triples, test_triples, headTailSelector, entity2id, id2entity, relation2id, id2relation

def get_relation2type_dict(train_triples, n_relations):
    relation2type = {}
    relation2head = {}
    relation2tail = {}

    for triple in train_triples:
        if triple[1] not in relation2head:
            relation2head[triple[1]] = [triple[0]]
            relation2tail[triple[1]] = [triple[2]]
        else:
            relation2head[triple[1]].append(triple[0])
            relation2tail[triple[1]].append(triple[2])

    train_d = get_all_true_triples(train_triples, {}, {})
    head_dict = train_d['head']
    tail_dict = train_d['tail']

    for i in range(n_relations):
        heads = relation2head[i]
        n2t_sum = 0
        for head in heads:
            n2t_sum = n2t_sum + len(tail_dict[(head, i)])
        n2t_avg = n2t_sum / len(heads)

        tails = relation2tail[i]
        t2n_sum = 0
        for tail in tails:
            t2n_sum = t2n_sum + len(head_dict[(i, tail)])
        t2n_avg = t2n_sum / len(tails)

        if n2t_avg > 1.5 and t2n_avg > 1.5:
            relation2type[i] = 0
        elif n2t_avg > 1.5 and t2n_avg <= 1.5:
            relation2type[i] = 2
        elif n2t_avg <= 1.5 and t2n_avg > 1.5:
            relation2type[i] = 1
        else:
            relation2type[i] = 3
    return relation2type

def get_init_embeddings(relinit, entinit):
    lstent = []
    lstrel = []
    with open(relinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstrel.append(tmp)
    with open(entinit) as f:
        for line in f:
            tmp = [float(val) for val in line.strip().split()]
            lstent.append(tmp)
    return np.array(lstent, dtype=np.float32), np.array(lstrel, dtype=np.float32)

def get_all_true_triples(train, valid, test):
    train = list(train.keys())
    valid = list(valid.keys())
    test = list(test.keys())
    all_triples = train + valid + test

    all_true_triples = {}
    head_triples = {}
    for triple in all_triples:
        tail_com = (triple[1], triple[2])
        if tail_com not in head_triples:
            head_triples[tail_com] = []
        head_triples[tail_com].append(triple[0])

    tail_triples = {}
    for triple in all_triples:
        head_com = (triple[0], triple[1])
        if head_com not in tail_triples:
            tail_triples[head_com] = []
        tail_triples[head_com].append(triple[2])

    all_true_triples['head'] = head_triples
    all_true_triples['tail'] = tail_triples

    return all_true_triples
