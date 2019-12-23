#  %%

import time
import torch
import matplotlib.pyplot as plt
from PIL import Image

from thinplatespline.tps import tps_warp
from thinplatespline.utils import (TOTEN, TOPIL,
        grid_points_2d, noisy_grid, grid_to_img)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

t0 = time.time()
img = Image.open("boris_johnson.jpg")
w, h = img.size
dense_grid = grid_points_2d(w, h, DEVICE)
X = grid_points_2d(7, 11, DEVICE)
Y = noisy_grid(7, 11, 0.15, DEVICE)
t1 = time.time()

print(f"time: {t1-t0:0.3f}", "created variables")

# %%

t0 = time.time()
warped_grid = tps_warp(X, Y, dense_grid)

ten_img = TOTEN(img)

ten_wrp = torch.grid_sampler_2d(
    ten_img[None, ...],
    warped_grid[None, None, :, :2],
    0, 0, False)[0, :, 0, :]

t1 = time.time()
print(f"time: {t1-t0:0.3f}", "warped image")

img_wrp = TOPIL(ten_wrp.reshape(3, h, w).cpu())

# %%

x1, y1 = grid_to_img(X, w, h)
x2, y2 = grid_to_img(Y, w, h)

fig, ax = plt.subplots(1, 2, figsize=[9, 7], sharey=True)
ax[0].imshow(img)
ax[0].plot(x1, y1, "+g", ms=15, mew=2, label="uniform")
ax[0].legend(loc=1)
ax[1].plot(x2, y2, "+r", ms=15, mew=2, label="target")
ax[1].imshow(img_wrp)
ax[1].legend(loc=1)
plt.tight_layout()
fig.savefig("plot.jpg", bbox_inches="tight")
