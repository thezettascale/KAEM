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
    'logs/uniform_RBF/DARCY_FLOW_1/generated_images.h5',
    'logs/uniform_FFT/DARCY_FLOW_1/generated_images.h5'
]

titles = ['RBF', 'Fourier']

images = []
for file_path in file_paths:
    with h5py.File(file_path, 'r') as h5_file:
        images.append(h5_file['samples'][()])

grid_size = (10, 10)  
fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(16, 8))

for dataset_idx, image_set in enumerate(images):
    for i in range(grid_size[0] * grid_size[1]):
        row, col = divmod(i, grid_size[1])
        col += dataset_idx * grid_size[1]  
        ax = axes[row, col]
        
        img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
        
        ax.imshow(img)
        ax.axis('off')  
    
    axes[0, dataset_idx * grid_size[1] + 4].set_title(titles[dataset_idx], fontsize=30, pad=10)

plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig('figures/results/RBFvsFFT.png')

