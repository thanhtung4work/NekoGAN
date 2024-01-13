import numpy as np

imgs = np.load("./data/full_numpy_bitmap_cat.npy")

max = np.max(imgs)
print(imgs.dtype)