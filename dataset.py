#coding=utf-8
import numpy as np
import scipy
import scipy.sparse
import pdb
# import torch
# import torch.utils.data
import mxnet
import os


class CFTrainDataset(mxnet.gluon.data.SimpleDataset):
    # torch.utils.data.dataset.Dataset
    def __init__(self, train_fname, nb_neg):
        self._load_train_matrix(train_fname) #input the training dataset file
        self.nb_neg = nb_neg   #? negative samples yes

    def _load_train_matrix(self, train_fname):
        def process_line(line):
            tmp = line.split('\t') #tab
            # user, item, rating
            return [int(tmp[0]), int(tmp[1]), float(tmp[2]) > 0]
        # these files are a few hundred megs tops
        # TODO: be unlazy? use pandas?
        with open(train_fname, 'r') as file: #打开train_fname,并且把其内容变量命名为file
            data = list(map(process_line, file))
        # train_fname = "ml-20m/train-ratings.csv"
        self.nb_users = max(data, key=lambda x: x[0])[0] + 1 # create implicit function
        self.nb_items = max(data, key=lambda x: x[1])[1] + 1
        self.data = list(filter(lambda x: x[2], data))
        self.mat = scipy.sparse.dok_matrix(
                (self.nb_users, self.nb_items), dtype=np.float32)
        for user, item, _ in data:
            self.mat[user, item] = 1.  # if interaction happened, the element in the matrix will be 1, otherwise be 0

    def __len__(self):
        return (self.nb_neg + 1) * len(self.data) # have no idea about nb_neg

    # use [] operator to get items
    def __getitem__(self, idx):
        if idx % (self.nb_neg + 1) == 0: #余数
            idx = idx // (self.nb_neg + 1)  #整数除法，这一步可以知道idx是nb_neg+1的多少倍
            return self.data[idx][0], self.data[idx][1], np.ones(1, dtype=np.float32)  # noqa: E501
        else:
            idx = idx // (self.nb_neg + 1)
            u = self.data[idx][0]
            j = np.random.randint(self.nb_items)
            while (u, j) in self.mat:
                j = np.random.randint(self.nb_items)
            return u, j, np.zeros(1, dtype=np.float32)

def load_test_ratings(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[0:2])
        return list(tmp)
    lines = open(fname, 'r').readlines()
    ratings = map(process_line, lines)
    return list(ratings)


def load_test_negs(fname):
    def process_line(line):
        tmp = map(int, line.split('\t')[1:])  # it means changing string to int
        return list(tmp)
    lines = open(fname, 'r').readlines()
    negs = map(process_line, lines)
    return list(negs)


def main():
    train_dataset = CFTrainDataset("ml-20m/train-ratings.csv", 10)
    print("finished")

if __name__ == '__main__':
    main()
