import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Compute Modern"],
        "axes.unicode_minus": False,
        "text.latex.preamble": (
            r"\usepackage{amsmath} "
            r"\usepackage{amsfonts} "
            r"\usepackage{amssymb} "
            r"\usepackage{bm} "
            r"\newcommand{\probP}{\text{I\kern-0.15em P}}"
        ),
    }
)

DATASETS = ["MNIST", "FMNIST"]
GRID_SIZES = [10, 20]
FNCS = ["RBF", "FFT"]

for dataset in DATASETS:
    for grid_size in GRID_SIZES:
        for fnc in FNCS:
            # File paths to HDF5 files
            file_paths = [
                f"logs/uniform_{fnc}/{dataset}_1/generated_images.h5",
                f"logs/lognormal_{fnc}/{dataset}_1/generated_images.h5",
                f"logs/gaussian_{fnc}/{dataset}_1/generated_images.h5",
            ]

            # titles = ['Gaussian', 'Lognormal', 'Uniform']
            titles = [
                r"$\mathcal{U}(\bm{z}; \; \bm{0}, \bm{1})$",
                r"$\text{Lognormal}(\bm{z}; \; \bm{0}, \bm{1})$",
                r"$\mathcal{N}(\bm{z}; \; \bm{0}, \bm{1})$",
            ]

            # Load generated images
            images = []
            for file_path in file_paths:
                with h5py.File(file_path, "r") as h5_file:
                    images.append(h5_file["samples"][()])

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
                        ax.imshow(images, cmap="gray_r")  # gray_r is inverted gray
                    else:
                        ax.imshow(images, cmap="gray")

                    ax.axis("off")

                axes[0, dataset_idx * grid_size + grid_size // 2].set_title(
                    titles[dataset_idx], fontsize=40, pad=10
                )

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(
                f"figures/results/{dataset}_priors_{fnc}_{grid_size}x{grid_size}.png"
            )
