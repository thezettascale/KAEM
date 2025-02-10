import h5py
import numpy as np
import matplotlib.pyplot as plt

h5_file = 'PDE_data/darcy_32/darcy_test_32.h5'

with h5py.File(h5_file, 'r') as f:
    x_data = f['x'][:]  
    y_data = f['y'][:] 

slice_idx = 1
x_slice = x_data[slice_idx, :, :]  
y_slice = y_data[slice_idx, :, :]  

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im1 = axes[0].imshow(x_slice, cmap='viridis', aspect='auto', interpolation='nearest')
axes[0].set_title('Permeability Field', fontsize=18, fontweight='bold')
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(y_slice, cmap='coolwarm', aspect='auto', interpolation='nearest')
axes[1].set_title('Flow Pressure', fontsize=18, fontweight='bold')
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('figures/test/darcy_slice.png', dpi=300, bbox_inches='tight')

plt.show()
