import h5py
import matplotlib.pyplot as plt
import numpy as np

# File paths to HDF5 files
file_paths = [
    "logs/Vanilla/MNIST/importance/uniform_RBF/univariate/generated_images.h5",
    "logs/Vanilla/FMNIST/importance/gaussian_RBF/univariate/generated_images.h5",
]

images = []
for file_path in file_paths:
    with h5py.File(file_path, "r") as h5_file:
        images.append(h5_file["samples"][()])

grid_size = (10, 10)
fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(16, 8))

for dataset_idx, image_set in enumerate(images):
    for i in range(grid_size[0] * grid_size[1]):
        row, col = divmod(i, grid_size[1])
        col += dataset_idx * grid_size[1]
        ax = axes[row, col]

        img = np.transpose(image_set[i, :, :, :], (1, 2, 0))

        ax.imshow(img, cmap="gray")
        ax.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig("figures/results/nist.png")
