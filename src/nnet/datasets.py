from os import listdir
from os.path import join
import pickle

import torch
from torch.utils import data


class nnetDataset(data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return self.data[index, :], self.label[index]


class nnetDatasetSeq(data.Dataset):

    def __init__(self, path):
        self.path = path
        with open(join(path, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)
        self.labels = torch.load(join(self.path, 'labels.pkl'))
        self.ids = list(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = torch.load(join(self.path, self.ids[index]))
        l = self.lengths[self.ids[index]]
        lab = self.labels[self.ids[index]]
        return x, l, lab


class nnetDataset3Seq(data.Dataset):

    def __init__(self, paths):
        self.path1 = paths[0]
        self.path2 = paths[1]
        self.path3 = paths[2]
        with open(join(self.path1, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)
        self.labels = torch.load(join(self.path1, 'labels.pkl'))
        self.ids = list(self.labels.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x1 = torch.load(join(self.path1, self.ids[index]))
        x2 = torch.load(join(self.path2, self.ids[index]))
        x3 = torch.load(join(self.path3, self.ids[index]))
        l = self.lengths[self.ids[index]]
        lab = self.labels[self.ids[index]]
        return x1, x2, x3, l, lab


class nnetDatasetSeqAE(data.Dataset):

    def __init__(self, path):
        self.path = path
        self.ids = [f for f in listdir(self.path) if f.endswith('.pt')]
        with open(join(path, 'lengths.pkl'), 'rb') as f:
            self.lengths = pickle.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        x = torch.load(join(self.path, self.ids[index]))
        l = self.lengths[self.ids[index]]
        return x, l
