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
formulas = [r"$\text{Lognormal}(\bm{z}; \; \bm{0}, \bm{1})$", r"$\mathcal{U}(\bm{z}; \; \bm{0}, \bm{1})$", r"$\mathcal{N}(\bm{z}; \; \bm{0}, \bm{1})$"]
bases = ['RBF', 'FFT']

grid_size = (3, 3)  
fig, axes = plt.subplots(grid_size[0] * 3, grid_size[1] * 3, figsize=(20, 20))  # Changed to 3 columns

for prior_idx, prior in enumerate(priors):
    # Load true images
    true_images_path = f'logs/{prior}_RBF/DARCY_FLOW_1/real_images.h5'
    with h5py.File(true_images_path, 'r') as h5_file:
        true_images = h5_file['samples'][()]

    # Load generated images
    file_paths = [
        f'logs/{prior}_RBF/DARCY_FLOW_1/generated_images.h5',
        f'logs/{prior}_FFT/DARCY_FLOW_1/generated_images.h5'
    ]

    images = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as h5_file:
            images.append(h5_file['samples'][()])

    # Plot true images
    for i in range(grid_size[0] * grid_size[1]):
        row = i // grid_size[1] + prior_idx * grid_size[0]
        col = i % grid_size[1]
        ax = axes[row, col]
        
        img = np.transpose(true_images[i, :, :, :], (1, 2, 0))
        ax.imshow(img)
        ax.axis('off')

    # Plot generated images
    for dataset_idx, image_set in enumerate(images):
        for i in range(grid_size[0] * grid_size[1]):
            row = i // grid_size[1] + prior_idx * grid_size[0]
            col = i % grid_size[1] + (dataset_idx + 1) * grid_size[1]
            ax = axes[row, col]
            
            img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
            ax.imshow(img)
            ax.axis('off')

    # Set titles for each row
    if prior_idx == 0:
        axes[0, grid_size[1] // 2].set_title('Ground Truth', fontsize=30, pad=20)
        axes[0, grid_size[1] + grid_size[1] // 2].set_title('RBF', fontsize=30, pad=20)
        axes[0, 2 * grid_size[1] + grid_size[1] // 2].set_title('Fourier', fontsize=30, pad=20)
    
    # Add prior type label
    axes[prior_idx * grid_size[0] + grid_size[0]//2, 0].text(
        1.2, 0.5, formulas[prior_idx],
        fontsize=30,
        rotation=45,
        transform=axes[prior_idx * grid_size[0] + grid_size[0]//2, -1].transAxes,
        verticalalignment='center'
    )

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f'figures/results/darcy_3x3.png', dpi=300)

