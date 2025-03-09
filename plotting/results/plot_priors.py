import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'text.usetex': True, 
    'font.family': 'serif', 
    'font.serif': ['Compute Modern'], 
    'axes.unicode_minus': False, 
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{bm} \newcommand{\probP}{\text{I\kern-0.15em P}}'  
})

# File paths to HDF5 files
file_paths = [
    'logs/gaussian_FFT/DARCY_FLOW_1/real_images.h5',
    'logs/gaussian_RBF/DARCY_FLOW_1/generated_images.h5',
    'logs/gaussian_FFT/DARCY_FLOW_1/generated_images.h5',
]

titles = ['True Samples', 'RBF', 'FFT']

# Load generated images
images = []
for file_path in file_paths:
    with h5py.File(file_path, 'r') as h5_file:
        images.append(h5_file['samples'][()])

grid_size = (13, 13)  
fig, axes = plt.subplots(grid_size[0], grid_size[1] * 3, figsize=(18, 6))

for dataset_idx, image_set in enumerate(images):
    for i in range(grid_size[0] * grid_size[1]):
        row, col = divmod(i, grid_size[1])
        col += dataset_idx * grid_size[1]  
        ax = axes[row, col]
        
        # Calculate error field
        images = np.transpose(image_set[i+60, :, :, :], (1, 2, 0))
        
        # Use inverted colormap for middle dataset
        if dataset_idx == 1:
            ax.imshow(images, cmap='cividis')  # gray_r is inverted gray
        else:
            ax.imshow(images, cmap='viridis')
            
        ax.axis('off')  
    
    axes[0, dataset_idx * grid_size[1] + 6].set_title(titles[dataset_idx], fontsize=40, pad=10)

plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig('figures/results/RBFvsFFT_gaussian_13x13.png')

