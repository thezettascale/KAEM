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


priors = ["_mixture", ""]
datasets = ["SVHN", "CIFAR10"]
GRID_SIZES = [10, 20]

for prior in priors:
    for dataset in datasets:
        for grid_size in GRID_SIZES:
            # MLE/ULA images
            mle_file_path = f'logs/Vanilla/n_z=100/ULA{prior}/cnn=true/{dataset}_1/generated_images.h5'
            # SE/ULA images
            se_file_path = f'logs/Thermodynamic{prior}/n_z=100/{dataset}_1/generated_images.h5'

            # Plot MLE/ULA grid
            with h5py.File(mle_file_path, 'r') as h5_file:
                mle_images = h5_file['samples'][()]
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            for i in range(grid_size * grid_size):
                row, col = divmod(i, grid_size)
                ax = axes[row, col]
                
                img = np.transpose(mle_images[i, :, :, :], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
            
            axes[0, grid_size // 2].set_title("MLE / ULA", fontsize=40, pad=10)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f'figures/results/{dataset}{prior}_mle_{grid_size}x{grid_size}.png')
            plt.close()

            # Plot SE/ULA grid
            with h5py.File(se_file_path, 'r') as h5_file:
                se_images = h5_file['samples'][()]
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
            for i in range(grid_size * grid_size):
                row, col = divmod(i, grid_size)
                ax = axes[row, col]
                
                img = np.transpose(se_images[i, :, :, :], (1, 2, 0))
                ax.imshow(img)
                ax.axis('off')
            
            axes[0, grid_size // 2].set_title("SE / ULA", fontsize=40, pad=10)
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(f'figures/results/{dataset}{prior}_se_{grid_size}x{grid_size}.png')
            plt.close()