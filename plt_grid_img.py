import h5py
import matplotlib.pyplot as plt

# File path to your HDF5 file
file_path = 'logs/Thermodynamic/uniform/MNIST_1/generated_images.h5'

with h5py.File(file_path, 'r') as h5_file:
    image_data = h5_file['samples'][()] 

# Define grid dimensions
grid_size = (10, 10) 
fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(8, 8))

for i in range(grid_size[0] * grid_size[1]):
    row, col = divmod(i, grid_size[1])
    ax = axes[row, col]
    
    img = image_data[:, :, i].reshape(28, 28)
    
    ax.imshow(img, cmap='gray')
    ax.axis('off')  

# plt.tight_layout()
plt.show()
