import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import Generator

LATENT_DIM = 300
IMG_SHAPE = (784, )

generator = Generator(LATENT_DIM, IMG_SHAPE)
generator.load_state_dict(torch.load("./weights/generator.pth"))

noise = torch.Tensor(np.random.normal(0, 1, (2, LATENT_DIM)))

imgs = generator(noise)
imgs = torch.reshape(imgs, (-1, 28, 28))
imgs = imgs.detach().numpy()

plt.imshow(imgs[0])
plt.show()