import argparse
import logging
import random
import json
import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from dataloader import TrainDataset, TestDataset
from args import parse_args
from tqdm import tqdm
from utils import *
from model import train_step, test_step, TransE
 # ConvKB, RotatE
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def main(args):
    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    set_logger(args)

    logging.info("Loading data...")

    train, valid, test, headTailSelector, \
    entity2id, id2entity, relation2id, id2relation = build_data(path=args.data_path)

    all_true_triples = get_all_true_triples(train, valid, test)
    train_true_triples = get_all_true_triples(train, {}, {})

    relation2type = get_relation2type_dict(train, len(id2relation.keys()))

    data_size = len(train)
    nentity = len(entity2id.keys())
    nrelation = len(relation2id.keys())

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)

    if args.model == "RotatE":
        kge_model = RotatE(
            model_name="",
            nentity=nentity,
            nrelation=nrelation,
            args=args
        )
    elif args.model == "ConvKB":
        kge_model = ConvKB(
            model_name="",
            nentity=nentity,
            nrelation=nrelation,
            args=args
        )
    elif args.model == "TransE":
        kge_model = TransE(
            model_name="",
            nentity=nentity,
            nrelation=nrelation,
            args=args
        )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()

    if args.do_train:
        # Set training dataloader iterator
        train_dataloader = DataLoader(
            TrainDataset(train, train_true_triples, headTailSelector, entity2id, id2entity, relation2id, id2relation, neg_ratio=args.neg_ratio),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num),
            collate_fn=TrainDataset.collate_fn,
            pin_memory=True
        )

        # Set training configuration
        current_learning_rate = args.learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, kge_model.parameters()), 
            lr=current_learning_rate, 
            weight_decay=0
        )
        # CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.5, last_epoch=-1)
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.use_init:
        hidden_size = 100
        if hidden_size != args.hidden_dim:
            raise Exception("init embedding not equal model embedding")

        init_entity_embs, init_relation_embs = get_init_embeddings(
            args.data_path + "/relation2vec"+str(hidden_size)+".init",
            args.data_path + "/entity2vec"+str(hidden_size)+".init")

        kge_model.load_embedding(init_entity_embs, init_relation_embs)


    if args.do_valid:
        valid_dataloader_head = DataLoader(
            TestDataset(
                valid, all_true_triples, entity2id, id2entity, \
                relation2id, id2relation, "head", args.eval_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2), 
            collate_fn=TestDataset.collate_fn,
            pin_memory=True
        )

        valid_dataloader_tail = DataLoader(
            TestDataset(
                valid, all_true_triples, entity2id, id2entity, \
                relation2id, id2relation, "tail", args.eval_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2), 
            collate_fn=TestDataset.collate_fn,
            pin_memory=True
        )

        valid_dataset_list = [valid_dataloader_head, valid_dataloader_tail]

        test_dataloader_head = DataLoader(
            TestDataset(
                test, all_true_triples, entity2id, id2entity, \
                relation2id, id2relation, "head", args.eval_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2), 
            collate_fn=TestDataset.collate_fn,
            pin_memory=True
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test, all_true_triples, entity2id, id2entity, \
                relation2id, id2relation, "tail", args.eval_type
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2), 
            collate_fn=TestDataset.collate_fn,
            pin_memory=True
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            # warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing Model...')
        init_step = 0

    step = init_step
    
    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('lmbda = %f' % args.lmbda)

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        max_val_value = 0
        training_logs = []
        step = 1
        scheduler_step = args.scheduler_step
        #Training Loop
        num_batches_per_epoch = int((data_size - 1) / args.batch_size) +1
        training_range = range(args.epochs)
        for epoch in training_range:
            train_iterator = iter(train_dataloader)
            for batch_num in range(num_batches_per_epoch):
                log = train_step(kge_model, optimizer, train_iterator, args)
                training_logs.append(log)

                if step % args.save_checkpoint_steps == 0:
                    save_variable_list = {
                        'step': step, 
                        'current_learning_rate': current_learning_rate,
                        'warm_up_steps': warm_up_steps
                    }
    #                 save_model(kge_model, optimizer, save_variable_list, args)

                if step % args.log_steps == 0:
                    metrics = {}
                    for metric in training_logs[0].keys():
                        metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                    log_metrics('Training average', step, metrics)
                    training_logs = []

                if step % scheduler_step == 0:
                    scheduler.step()

                if args.do_valid and step % args.valid_steps == 0:
                    logging.info('Evaluating on Valid Dataset...')
                    # set_trace()
                    metrics = test_step(kge_model, valid, valid_dataset_list, \
                                                  entity2id, id2entity, relation2id, id2relation, relation2type, args)
                    if metrics['MRR'] > max_val_value:
                        log_metrics('Valid', step, metrics)
                        save_variable_list = {
                            'step': step, 
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        save_model(kge_model, optimizer, save_variable_list, args, True)
                        logging.info("Max MRR, now evaluate test dataset")
                        # metrics = test_step(kge_model, test, test_dataset_list, \
                                                        # entity2id, id2entity, relation2id, id2relation, relation2type, args)
                        # log_metrics('Test', step, metrics)
                    else:
                        log_metrics('Valid', step, metrics)
                step += 1

        save_variable_list = {
            'step': step, 
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
#         save_model(kge_model, optimizer, save_variable_list, args)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = test_step(kge_model, test, test_dataset_list, \
                                        entity2id, id2entity, relation2id, id2relation, relation2type, args)
        log_metrics('Test', step, metrics)

if __name__ == '__main__':
    main(parse_args())
