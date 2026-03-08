import h5py
import matplotlib.pyplot as plt
import numpy as np

file_path_real = "logs/Baseline/CELEBA/VAE/generated_images_epoch_60.h5"
file_path_generated = (
    "logs/Thermodynamic/CELEBA/ULA/mixture/generated_images_epoch_60.h5"
)

with h5py.File(file_path_real, "r") as h5_file:
    real_data = h5_file["samples"][()]

with h5py.File(file_path_generated, "r") as h5_file:
    generated_data = h5_file["samples"][()]

grid_size = (7, 7)
fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(grid_size[0], grid_size[1] * 2 + 1, wspace=0, hspace=0)

for i in range(grid_size[0] * grid_size[1]):
    row, col = divmod(i, grid_size[1])
    ax = fig.add_subplot(gs[row, col])
    img = np.transpose(real_data[i, :, :, :], (1, 2, 0))
    ax.imshow(img)
    ax.axis("off")

for i in range(grid_size[0] * grid_size[1]):
    row, col = divmod(i, grid_size[1])
    ax = fig.add_subplot(gs[row, col + grid_size[1] + 1])
    img = np.transpose(generated_data[i, :, :, :], (1, 2, 0))
    ax.imshow(img)
    ax.axis("off")

plt.savefig("garbage/grid.png", bbox_inches="tight", pad_inches=0.2, dpi=700)
plt.show()
