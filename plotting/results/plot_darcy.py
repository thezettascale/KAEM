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
# prior = 'lognormal'
priors = ["lognormal", "uniform", "gaussian"]
bases = ['RBF', 'FFT']

grid_size = (10, 10)  
fig, axes = plt.subplots(grid_size[0] * 3, grid_size[1] * 2, figsize=(16, 8))

for prior in priors:
    file_paths = [
        f'logs/{prior}_RBF/DARCY_FLOW_1/generated_images.h5',
        f'logs/{prior}_FFT/DARCY_FLOW_1/generated_images.h5'
    ]

    images = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5_file:
            images.append(h5_file['samples'][()])

    for dataset_idx, image_set in enumerate(images):
        for i in range(grid_size[0] * grid_size[1]):
            row, col = divmod(i, grid_size[1])
            col += dataset_idx * grid_size[1]  
            ax = axes[row, col]
            
            img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
            ax.imshow(img)
            ax.axis('off')  
        
        # Set title for each grid
        axes[0, dataset_idx * grid_size[1] + 4].set_title(bases[dataset_idx], fontsize=12, pad=20)
    
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()
# plt.savefig(f'figures/results/darcy_3x2.png')

