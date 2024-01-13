import numpy as np
from torch import nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            *self._init_gen_block(latent_dim, 128, False),
            *self._init_gen_block(128, 256),
            *self._init_gen_block(256, 512),
            *self._init_gen_block(512, 1024),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh()
        )

    def _init_gen_block(self, in_feat: int, out_feat: int, normalized: bool = True):
        layers = [nn.Linear(in_feat, out_feat)]
        if normalized:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.ReLU(inplace=True))
        return layers
    
    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), *self.img_shape)
        return img
