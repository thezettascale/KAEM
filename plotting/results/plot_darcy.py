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

# File paths to HDF5 files
# prior = 'lognormal'
priors = ["lognormal", "uniform", "gaussian", "ebm"]
formulas = [
    r"$\text{Lognormal}(\bm{z}; \; \bm{0}, \bm{1})$",
    r"$\mathcal{U}(\bm{z}; \; \bm{0}, \bm{1})$",
    r"$\mathcal{N}(\bm{z}; \; \bm{0}, \bm{1})$",
]
bases = ["RBF", "FFT"]

grid_size = (3, 3)
fig, axes = plt.subplots(
    grid_size[0] * 3,
    grid_size[1] * 3,
    figsize=(30, 30),
)  # Changed to 3 columns

for prior_idx, prior in enumerate(priors):
    # Load true images
    true_images_path = f"logs/Vanilla/DARCY_FLOW/importance/{prior}_RBF/univariate/real_images.h5"
    with h5py.File(true_images_path, "r") as h5_file:
        true_images = h5_file["samples"][()]

    # Load generated images
    file_paths = [
        f"logs/Vanilla/DARCY_FLOW/importance/{prior}_RBF/univariate/generated_images.h5",
        f"logs/Vanilla/DARCY_FLOW/importance/{prior}_FFT/univariate/generated_images.h5",
    ]

    images = []
    for file_path in file_paths:
        with h5py.File(file_path, "r") as h5_file:
            images.append(h5_file["samples"][()])

    # Plot true images
    for i in range(grid_size[0] * grid_size[1]):
        row = i // grid_size[1] + prior_idx * grid_size[0]
        col = i % grid_size[1]
        ax = axes[row, col]

        img = np.transpose(true_images[i, :, :, :], (1, 2, 0))
        ax.imshow(img)
        ax.axis("off")

    # Plot generated images
    for dataset_idx, image_set in enumerate(images):
        for i in range(grid_size[0] * grid_size[1]):
            row = i // grid_size[1] + prior_idx * grid_size[0]
            col = i % grid_size[1] + (dataset_idx + 1) * grid_size[1]
            ax = axes[row, col]

            img = np.transpose(image_set[i, :, :, :], (1, 2, 0))
            if dataset_idx == 0:
                ax.imshow(img, cmap="cividis")
            else:
                ax.imshow(img)
            ax.axis("off")

    # Set titles for each row
    if prior_idx == 0:
        axes[0, grid_size[1] // 2].set_title("Ground Truth", fontsize=44, pad=20)
        axes[0, grid_size[1] + grid_size[1] // 2].set_title("RBF", fontsize=44, pad=20)
        axes[0, 2 * grid_size[1] + grid_size[1] // 2].set_title(
            "Fourier", fontsize=44, pad=20
        )

    # Add prior type label
    axes[prior_idx * grid_size[0] + grid_size[0] // 4, 0].text(
        -0,
        0.5,
        formulas[prior_idx],
        fontsize=44,
        rotation=45,
        transform=axes[prior_idx * grid_size[0] + grid_size[0] // 4, 0].transAxes,
        verticalalignment="center",
        horizontalalignment="right",
    )

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig("figures/results/darcy_3x3.png", dpi=300)
