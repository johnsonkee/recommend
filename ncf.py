import os
import heapq
import math
import time
from functools import partial
from datetime import datetime
from collections import OrderedDict
from argparse import ArgumentParser
import pdb


import tqdm
import numpy as np
# import torch
# import torch.nn as nn
# from torch import multiprocessing as mp

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet.gluon import nn
import multiprocessing as mp


import utils
from neumf import NeuMF
from dataset import CFTrainDataset, load_test_ratings, load_test_negs
from convert import (TEST_NEG_FILENAME, TEST_RATINGS_FILENAME,
                     TRAIN_RATINGS_FILENAME)


def parse_args():
    parser = ArgumentParser(description="Train a Nerual Collaborative"
                                        " Filtering model")
    parser.add_argument('data', type=str,default='ml-20m',
                        help='path to test and training data files')
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='number of epochs for training')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='number of examples for each iteration')
    parser.add_argument('-f', '--factors', type=int, default=8,
                        help='number of predictive factors')
    parser.add_argument('--layers', nargs='+', type=int,
                        default=[64, 32, 16, 8],
                        help='size of hidden layers for MLP')
                        # TODO: change the negs's default nb
    parser.add_argument('-n', '--negative-samples', type=int, default=4,
                        help='number of negative examples per interaction')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('-k', '--topk', type=int, default=10,
                        help='rank for test examples to be considered a hit')
    parser.add_argument('--no-cuda', action='store_true',
                        help='use available GPUs')
    parser.add_argument('--seed', '-s', type=int,
                        help='manually set random seed for torch')
    parser.add_argument('--threshold', '-t', type=float,
                        help='stop training early at threshold')
    parser.add_argument('--processes', '-p', type=int, default=2,
                        help='Number of processes for evaluating model')
    return parser.parse_args()


def predict(model, users, items, ctx, batch_size=1024, use_cuda=True):
    batches = [(users[i:i + batch_size], items[i:i + batch_size])
               for i in range(0, len(users), batch_size)]
    preds = []
    #
    for user, item in batches:
        def proc(x):
            # convert numpy'ndarray to mxnet.NDArray
            x = nd.array(x)
            if use_cuda:
                x = x.as_in_context(ctx)# todo
            return x
            # TODO: data dimension is not suitable for the network
        outp = model(proc(user), proc(item), True)
        outp = outp.as_in_context(ctx).asnumpy()
        preds += list(outp.flatten())
    return preds


def _calculate_hit(ranked, test_item):
    return int(test_item in ranked)


def _calculate_ndcg(ranked, test_item):
    for i, item in enumerate(ranked):
        if item == test_item:
            return math.log(2) / math.log(i + 2)
    return 0.


def eval_one(rating, items, model, K, ctx,use_cuda=True):
    user = rating[0]
    test_item = rating[1]
    items.append(test_item)
    users = [user] * len(items)
    predictions = predict(model, users, items, use_cuda=use_cuda, ctx=ctx)

    map_item_score = {item: pred for item, pred in zip(items, predictions)}
    ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    hit = _calculate_hit(ranked, test_item)
    ndcg = _calculate_ndcg(ranked, test_item)
    return hit, ndcg


# K meand topk
def val_epoch(model, ratings, negs, K, ctx, use_cuda=True, output=None, epoch=None,
              processes=1):
    if epoch is None:
        print("Initial evaluation")
    else:
        print("Epoch {} evaluation".format(epoch))
    start = datetime.now()
    # model.eval()

    if processes > 1:

        context = mp.get_context('spawn')

        _eval_one = partial(eval_one, model=model, K=K, use_cuda=use_cuda, ctx=ctx)
        with context.Pool(processes=processes) as workers:
            hits_and_ndcg = workers.starmap(_eval_one, zip(ratings, negs))
        hits, ndcgs = zip(*hits_and_ndcg)
    else:
        hits, ndcgs = [], []
        for rating, items in zip(ratings, negs):
            hit, ndcg = eval_one(rating, items, model, K, use_cuda=use_cuda, ctx=ctx)
            hits.append(hit)
            ndcgs.append(ndcg)

    hits = np.array(hits, dtype=np.float32)
    ndcgs = np.array(ndcgs, dtype=np.float32)

    end = datetime.now()
    if output is not None:
        result = OrderedDict()
        result['timestamp'] = datetime.now()
        result['duration'] = end - start
        result['epoch'] = epoch
        result['K'] = K
        result['hit_rate'] = np.mean(hits)
        result['NDCG'] = np.mean(ndcgs)
        utils.save_result(result, output)

    return hits, ndcgs


def main():
    args = parse_args()
    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        # torch.manual_seed(args.seed)
        mx.random.seed(seed_state=args.seed)
        np.random.seed(seed=args.seed)

    # Save configuration to file
    config = {k: v for k, v in args.__dict__.items()}
    config['timestamp'] = "{:.0f}".format(datetime.utcnow().timestamp())
    config['local_timestamp'] = str(datetime.now())
    run_dir = "./run/neumf/{}".format(config['timestamp'])
    print("Saving config and results to {}".format(run_dir))
    if not os.path.exists(run_dir) and run_dir != '':
        os.makedirs(run_dir)
    utils.save_config(config, run_dir)   #defined in utils.py

    # Check that GPUs are actually available
    use_cuda = not args.no_cuda and mx.test_utils.list_gpus()

    t1 = time.time()
    # Load Data
    print('Loading data')
    train_dataset = CFTrainDataset(
        os.path.join(args.data, TRAIN_RATINGS_FILENAME), args.negative_samples)
    #in original file, use 8 core as defaul

    # shuffle means random the samples
    train_dataloader = mx.gluon.data.DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=8)

    test_ratings = load_test_ratings(os.path.join(args.data, TEST_RATINGS_FILENAME))  # noqa: E501
    test_negs = load_test_negs(os.path.join(args.data, TEST_NEG_FILENAME))
    nb_users, nb_items = train_dataset.nb_users, train_dataset.nb_items

    print('Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d'
          % (time.time()-t1, nb_users, nb_items, train_dataset.mat.nnz,
             len(test_ratings)))

    if(use_cuda):
        ctx = mx.gpu(0) # default to use NO.1 gpu can use docker to select a nvidia
    else:
        ctx = mx.cpu(0)

    # Create model
    model = NeuMF(nb_users, nb_items,
                  mf_dim=args.factors, mf_reg=0.,
                  mlp_layer_sizes=args.layers,
                  mlp_layer_regs=[0. for i in args.layers],
                  ctx=ctx)
    model.initialize(ctx=ctx)
    print(model)
    # todo 9: to change the function in utils
    # print("{} parameters".format(utils.count_parameters(model)))

    # model.collect_params()
    # Save model text description
    with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
        file.write(str(model))
    # model.save_parameters(os.path.join("/home/net.params", 'net.params'))

    # Create files for tracking training
    valid_results_file = os.path.join(run_dir, 'valid_results.csv')

    # Calculate initial Hit Ratio and NDCG
    hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                             use_cuda=use_cuda, processes=args.processes, ctx=ctx)
    print('Initial HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f}'
           .format(K=args.topk, hit_rate=np.mean(hits), ndcg=np.mean(ndcgs)))

############# hyperparameters
# Add optimizer and loss to graph
    lr = args.learning_rate
    bs = args.batch_size

    trainer = mx.gluon.Trainer(model.collect_params(),'adam',{'learning_rate': lr})
    mxnet_criterion = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()   # equivalent to lossfunction


    # training
    for epoch in range(args.epochs):
        begin = time.time()
        # tqdm shows the percentage of the process
        loader = tqdm.tqdm(train_dataloader)
        pdb.set_trace()
        for batch_index, (user, item, label) in enumerate(loader):
            # TODO 7: search the autograd in mxnet
            # todo : let user act in gpu
            user = nd.array(user,ctx=ctx)
            item = nd.array(item,ctx=ctx)
            label = nd.array(label,ctx=ctx)
            pdb.set_trace()

            # compute the gradient automatically
            with autograd.record():
                outputs = model(user, item)
                loss = mxnet_criterion(outputs, label.T)

            loss.backward()
            trainer.step(bs)

            # Save stats to file
            # description = ('Epoch {} Average Loss {loss:.4f}'
            #                .format(epoch, loss=loss.mean()))
            # loader.set_description(description)

        train_time = time.time() - begin
        begin = time.time()
        hits, ndcgs = val_epoch(model, test_ratings, test_negs, args.topk,
                                use_cuda=use_cuda, output=valid_results_file,
                                epoch=epoch, processes=args.processes, ctx=ctx)
        val_time = time.time() - begin
        print('Epoch {epoch}: HR@{K} = {hit_rate:.4f}, NDCG@{K} = {ndcg:.4f},'
              ' train_time = {train_time:.2f}, val_time = {val_time:.2f}'
              .format(epoch=epoch, K=args.topk, hit_rate=np.mean(hits),
                      ndcg=np.mean(ndcgs), train_time=train_time,
                      val_time=val_time))
        if args.threshold is not None:
            if np.mean(hits) >= args.threshold:
                print("Hit threshold of {}".format(args.threshold))
                # Save model text description after modelling
                run_dir = "./run/neumf/{}".format(config['timestamp'])
                with open(os.path.join(run_dir, 'model.txt'), 'w') as file:
                    file.write(str(model))
                # model.save_parameters(os.path.join("/home/net.params",'net.params'))
                return 0

if __name__ == '__main__':
    main()
