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

grid_size = (12, 12)  
fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(16, 8))

for dataset_idx, image_set in enumerate(images):
    for i in range(grid_size[0] * grid_size[1]):
        row, col = divmod(i, grid_size[1])
        col += dataset_idx * grid_size[1]  
        ax = axes[row, col]
        
        img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
        
        ax.imshow(img, cmap='gray')
        ax.axis('off')  

plt.subplots_adjust(wspace=0, hspace=0)
# plt.show()
plt.savefig('figures/results/nist.png')

# Does not make sense to use FID/KID for MNIST/FMNIST

# import os
# import h5py
# import torch
# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from torch_fidelity import calculate_metrics
# import matplotlib.pyplot as plt
# from PIL import Image
# import tempfile

# plt.rcParams.update({
#     'text.usetex': True, 
#     'font.family': 'serif', 
#     'font.serif': ['Compute Modern'], 
#     'axes.unicode_minus': False, 
#     'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb} \usepackage{bm} \newcommand{\probP}{\text{I\kern-0.15em P}}'  
# })

# # File paths to HDF5 files
# file_paths = [
#     ('logs/Vanilla/importance/MNIST_1/generated_images.h5', 'logs/Vanilla/importance/MNIST_1/real_images.h5'),
#     ('logs/Vanilla/importance/FMNIST_1/generated_images.h5', 'logs/Vanilla/importance/FMNIST_1/real_images.h5')
# ]

# batch_sizes = [1000, 1400]#, 1800, 2200, 2600, 3000]

# def load_images(file_path):
#     with h5py.File(file_path, 'r') as f:
#         images = np.array(f['samples'])
#     images = torch.tensor(images).repeat(1, 3, 1, 1)
#     return images

# def save_images_to_directory(images, directory):
#     os.makedirs(directory, exist_ok=True)
#     for i, img_tensor in enumerate(images):
#         img = img_tensor.permute(1, 2, 0).numpy() * 255
#         img = img.astype(np.uint8)
#         Image.fromarray(img).save(os.path.join(directory, f'image_{i}.png'))

# metrics_results = []

# grid_size = (12, 12)  
# fig, axes = plt.subplots(grid_size[0], grid_size[1] * 2, figsize=(12, 6))

# for dataset_idx, (gen_file_path, real_file_path) in enumerate(file_paths):
#     real_images = load_images(real_file_path)
#     gen_images = load_images(gen_file_path)
    
#     with tempfile.TemporaryDirectory() as real_images_dir:
#         save_images_to_directory(real_images, real_images_dir)
#         fids = []
#         kids = []
        
#         for batch_size in batch_sizes:
#             indices = np.random.choice(len(gen_images), batch_size, replace=False)
#             with tempfile.TemporaryDirectory() as gen_images_dir:
#                 save_images_to_directory(gen_images[indices], gen_images_dir)
#                 metrics = calculate_metrics(
#                     input1=real_images_dir,
#                     input2=gen_images_dir,
#                     fid=True, kid=True
#                 )
#                 fids.append(metrics['frechet_inception_distance'])
#                 kids.append(metrics['kernel_inception_distance_mean'])
        
#         inverse_batch_sizes = 1 / np.array(batch_sizes).reshape(-1, 1)
#         fid_reg = LinearRegression().fit(inverse_batch_sizes, np.array(fids).reshape(-1, 1))
#         kid_reg = LinearRegression().fit(inverse_batch_sizes, np.array(kids).reshape(-1, 1))
        
#         fid_infinity = fid_reg.predict(np.array([[0]]))[0, 0]
#         kid_infinity = kid_reg.predict(np.array([[0]]))[0, 0]
#         metrics_results.append((fid_infinity, kid_infinity))
        
#         for i in range(grid_size[0] * grid_size[1]):
#             row, col = divmod(i, grid_size[1])
#             col += dataset_idx * grid_size[1]  
#             ax = axes[row, col]
#             img = np.transpose(gen_images[i, :, :, :], (1, 2, 0))
#             ax.imshow(img, cmap='gray')
#             ax.axis('off')  

# for idx, (fid, kid) in enumerate(metrics_results):
#     col = idx * (grid_size[1] // 2)
#     axes[0, col].set_title(r'$\overline{\text{FID}}_\infty$: ' + f'{fid:.2f}', fontsize=18)
#     axes[0, col + 1].set_title(r'$\overline{\text{KID}}_\infty$: ' + f'{kid:.2f}', fontsize=18)

# plt.subplots_adjust(wspace=0, hspace=0)
# plt.savefig('figures/results/nist.png')
