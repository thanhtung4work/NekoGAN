import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data

from dataset import Dataset
from modules import Generator, Discriminator

IMG_SHAPE = (784, )
BATCH_SIZE = 64
LATENT_DIM = 300
LR = 0.001
N_EPOCHS = 10

dataloader = data.DataLoader(
    dataset = Dataset("./data/full_numpy_bitmap_cat.npy"),
    batch_size = BATCH_SIZE,
    shuffle = True
)

generator = Generator(LATENT_DIM, IMG_SHAPE)
discriminator = Discriminator(IMG_SHAPE)

try:
    generator.load_state_dict(torch.load("./weights/generator.pth"))
    discriminator.load_state_dict(torch.load("./weights/discriminator.pth"))
except:
    print("Cannot find weights!")

optimizer_G = optim.Adam(generator.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

# Loss function
adversarial_loss = torch.nn.BCELoss()

for epoch in range(N_EPOCHS):
    start = time.time()
    for i, batch in enumerate(dataloader):
        truth = torch.Tensor(batch.size(0), 1).fill_(1.0)
        fake = torch.Tensor(batch.size(0), 1).fill_(0.0)

        # ===============
        # Train Generator
        # ===============

        optimizer_G.zero_grad()

        # Sample noise
        noise = torch.Tensor(np.random.normal(0, 1, (batch.shape[0], LATENT_DIM)))

        # Generate batch of image
        gen_batch = generator(noise)

        # Loss 
        adv_loss = adversarial_loss(discriminator(gen_batch), truth)
        adv_loss.backward()
        optimizer_G.step()

        # ===================
        # Train Discriminator
        # ===================

        optimizer_D.zero_grad()
        
        real_loss = adversarial_loss(discriminator(batch), truth)
        fake_loss = adversarial_loss(discriminator(gen_batch.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    end = time.time()
    print(
        f"epoch: {epoch+1} ({end-start:.2f}s) | g_loss: {adv_loss.item():.3f} | d_loss: {d_loss.item():.3f}"
    )

torch.save(generator.state_dict(), "./weights/generator.pth")
torch.save(discriminator.state_dict(), "./weights/discriminator.pth")