import numpy as np
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat), 
            nn.ReLU(inplace=True),
            nn.Linear(in_feat, in_feat),
            nn.BatchNorm1d(in_feat),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        r = self.block(x)
        return self.relu(r + x)



class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__() 

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 256),
            nn.ReLU(inplace=True),
            ResBlock(256),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            ResBlock(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        flatten_img = img.view(img.size(0), -1)
        prob = self.model(flatten_img)
        return prob
