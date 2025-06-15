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

for prior in priors:
    for dataset in datasets:
        gen_file_path = f'logs/Vanilla/n_z=100/ULA{prior}/cnn=true/{dataset}_1/generated_images.h5'
        real_file_path = f'logs/Thermodynamic{prior}/n_z=100/{dataset}_1/generated_images.h5'

        titles = ["MLE / ULA", "SE / ULA"]
        images = []
        with h5py.File(gen_file_path, 'r') as h5_file:
            images.append(h5_file['samples'][()])
        with h5py.File(real_file_path, 'r') as h5_file:
            images.append(h5_file['samples'][()])
        
        grid_size = (7, 7)  
        fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(16, 8))
        
        for dataset_idx, image_set in enumerate(images):
            for i in range(grid_size[0] * grid_size[1]):
                row, col = divmod(i, grid_size[1])
                col += dataset_idx * grid_size[1]  
                ax = axes[row, col]
                
                img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
                
                ax.imshow(img)
                ax.axis('off')
            
            axes[0, dataset_idx * grid_size[0] + grid_size[0] // 2].set_title(titles[dataset_idx], fontsize=40, pad=10)

        # Add spacing between the two image sets (after 5th column)
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(grid_size[0]):
            axes[i, 4].set_axis_off()
            axes[i, 5].set_axis_off()

        plt.savefig(f'figures/results/{dataset}_{prior}.png')
        # plt.show()
        plt.close()