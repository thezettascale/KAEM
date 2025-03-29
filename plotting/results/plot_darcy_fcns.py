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

PRIORS = ["uniform", "lognormal", "gaussian"]
GRID_SIZES = [7, 13]

for prior in PRIORS:
    for grid_size in GRID_SIZES:
        # File paths to HDF5 files
        file_paths = [
            f'logs/{prior}_RBF/DARCY_FLOW_1/real_images.h5',
            f'logs/{prior}_RBF/DARCY_FLOW_1/generated_images.h5',
            f'logs/{prior}_FFT/DARCY_FLOW_1/generated_images.h5',
        ]

        titles = ['True Sample', 'RBF', 'FFT']

        # Load generated images
        images = []
        for file_path in file_paths:
            with h5py.File(file_path, 'r') as h5_file:
                images.append(h5_file['samples'][()])

        fig, axes = plt.subplots(grid_size, grid_size * 3, figsize=(18, 6))

        for dataset_idx, image_set in enumerate(images):
            for i in range(grid_size * grid_size):
                row, col = divmod(i, grid_size)
                col += dataset_idx * grid_size  
                ax = axes[row, col]
                
                # Calculate error field
                images = np.transpose(image_set[i, :, :, :], (1, 2, 0))
                
                # Use inverted colormap for middle dataset
                if dataset_idx == 1:
                    ax.imshow(images, cmap='cividis')  
                else:
                    ax.imshow(images, cmap='viridis')
                    
                ax.axis('off')  
            
            axes[0, dataset_idx * grid_size + grid_size // 2].set_title(titles[dataset_idx], fontsize=40, pad=10)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(f'figures/results/darcy_fncs_{prior}_{grid_size}x{grid_size}.png')

