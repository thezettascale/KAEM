import h5py
import matplotlib.pyplot as plt
import numpy as np

# File paths to HDF5 files
file_paths = [
    'logs/Vanilla/importance/MNIST_1/generated_images.h5',
    'logs/Vanilla/importance/FMNIST_1/generated_images.h5'
]

images = []
for file_path in file_paths:
    with h5py.File(file_path, 'r') as h5_file:
        images.append(h5_file['samples'][()])

grid_size = (15, 15)  
fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(16, 8))

for dataset_idx, image_set in enumerate(images):
    for i in range(grid_size[0] * grid_size[1]):
        row, col = divmod(i, grid_size[1])
        col += dataset_idx * grid_size[1]  
        ax = axes[row, col]
        
        img = image_set[i, :, :, :].reshape(28, 28, 1)
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')  

plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig('figures/results/nist.png')
