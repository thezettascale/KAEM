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
GRID_SIZES = [5, 10]
FNCS = ["RBF", "FFT"]

for dataset in DATASETS:
    for grid_size in GRID_SIZES:
        for fnc in FNCS:
            # File paths to HDF5 files
            file_paths = [
                f"logs/Vanilla/{dataset}/importance/ebm_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/ebm_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/gaussian_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/gaussian_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/lognormal_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/lognormal_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/uniform_{fnc}/univariate/generated_images.h5",
                f"logs/Vanilla/{dataset}/importance/uniform_{fnc}/univariate/generated_images.h5",
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

            fig, axes = plt.subplots(grid_size, grid_size * 8, figsize=(32, 4))

            for dataset_idx, image_set in enumerate(images):
                for i in range(grid_size * grid_size):
                    row, col = divmod(i, grid_size)
                    col += dataset_idx * grid_size
                    ax = axes[row, col]

                    # Calculate error field
                    images = np.transpose(image_set[i, :, :, :], (1, 2, 0))

                    # Use inverted colormap for middle dataset
                    if (
                        dataset_idx == 1
                        or dataset_idx == 3
                        or dataset_idx == 5
                        or dataset_idx == 7
                    ):
                        ax.imshow(images, cmap="gray_r")  # gray_r is inverted gray
                    else:
                        ax.imshow(images, cmap="gray")

                    ax.axis("off")

                axes[0, 0].set_title("EBM-FFT", fontsize=24)
                axes[0, 1].set_title("EBM-RBF", fontsize=24)
                axes[0, 2].set_title("Gaussian-FFT", fontsize=24)
                axes[0, 3].set_title("Gaussian-RBF", fontsize=24)
                axes[0, 4].set_title("Lognormal-FFT", fontsize=24)
                axes[0, 5].set_title("Lognormal-RBF", fontsize=24)
                axes[0, 6].set_title("Uniform-FFT", fontsize=24)
                axes[0, 7].set_title("Uniform-RBF", fontsize=24)

            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(
                f"figures/results/{dataset.lower()}_{fnc.lower()}_{grid_size}x{grid_size}.png",
                dpi=300,
                bbox_inches="tight",
            )
