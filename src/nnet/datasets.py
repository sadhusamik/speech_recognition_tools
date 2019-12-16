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

