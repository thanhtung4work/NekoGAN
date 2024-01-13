import numpy as np
import torch
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, np_file):
        self.data = np.load(np_file).astype('float32') / 255.0
        self.data = torch.from_numpy(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
