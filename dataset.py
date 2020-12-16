from torch.utils.data import Dataset
import numpy as np
import os.path as osp
import torch


class BinaryDataset(Dataset):
    def __init__(self, data_root, phase):
        self.records = np.load(osp.join(data_root, phase, "sample.npy"))

    def __getitem__(self, index):
        x = self.records[index].astype('float64')
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def __len__(self):
        return len(self.records)
