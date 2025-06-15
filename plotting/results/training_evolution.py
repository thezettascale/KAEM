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

save_name = "figures/results/svhn_mle_evol.png"
file_path = "logs/Vanilla/n_z=100/ULA/cnn=true/SVHN_1"

epochs = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

grid_size = (3, 3)
fig, axes = plt.subplots(grid_size[0], grid_size[1] * len(epochs), figsize=(6 * len(epochs), 6))

for epoch_idx, epoch in enumerate(epochs):
    file = f"{file_path}/generated_images_epoch_{epoch}.h5"
    with h5py.File(file, 'r') as h5_file:
        images = h5_file['samples'][()]
        
        for i in range(grid_size[0] * grid_size[1]):
            row, col = divmod(i, grid_size[1])
            col += epoch_idx * grid_size[1]
            ax = axes[row, col]
            
            img = np.transpose(images[i, :, :, :], (1, 2, 0))
            ax.imshow(img)
            ax.axis('off')
        
        axes[0, epoch_idx * grid_size[0] + grid_size[0] // 2].set_title(f"Epoch {epoch}", fontsize=40, pad=10)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(save_name)
# plt.show()
plt.close()
