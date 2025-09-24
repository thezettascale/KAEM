import os

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

DATASETS = {
    "DARCY_FLOW": {"grid_size": 20, "cmap": "viridis"},
    "MNIST": {"grid_size": 20, "cmap": "gray"},
    "FMNIST": {"grid_size": 20, "cmap": "gray"},
}

PRIORS = ["uniform", "lognormal", "gaussian", "ebm"]
FUNCTIONS = ["RBF", "FFT"]

output_dir = "figures/results/individual_plots"
os.makedirs(output_dir, exist_ok=True)


def plot_prior_function_grid(dataset, prior, function, grid_size, cmap):
    """Generate and save a single plot for a specific prior-function combination."""

    # Only load generated images for individual plots
    gen_path = f"logs/Vanilla/{dataset}/importance/{prior}_{function}/univariate/generated_images.h5"

    try:
        with h5py.File(gen_path, "r") as h5_file:
            generated_images = h5_file["samples"][()]

        # Create square grid for generated images only
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

        if grid_size == 1:
            axes = axes.reshape(1, 1)

        # Plot generated images in square grid
        for i in range(grid_size * grid_size):
            row, col = divmod(i, grid_size)
            ax = axes[row, col]

            img = np.transpose(generated_images[i, :, :, :], (1, 2, 0))
            ax.imshow(img, cmap=cmap)
            ax.axis("off")

        prior_labels = {
            "uniform": r"$\mathcal{U}(\bm{z}; \; \bm{0}, \bm{1})$",
            "lognormal": r"$\text{Lognormal}(\bm{z}; \; \bm{0}, \bm{1})$",
            "gaussian": r"$\mathcal{N}(\bm{z}; \; \bm{0}, \bm{1})$",
            "ebm": "EBM",
        }

        fig.suptitle(
            f"{dataset} - {prior_labels[prior]} - {function}", fontsize=18, y=0.95
        )

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        filename = f"{dataset.lower()}_{prior}_{function.lower()}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    except FileNotFoundError:
        print(f"Warning: Could not find data for {dataset} - {prior} - {function}")
    except Exception as e:
        print(f"Error processing {dataset} - {prior} - {function}: {e}")


def plot_real_images_reference(dataset, grid_size, cmap):
    """Generate a reference plot with real images for each dataset."""

    # Try to find real images from any prior/function combination
    for prior in PRIORS:
        for function in FUNCTIONS:
            real_path = f"logs/Vanilla/{dataset}/importance/{prior}_{function}/univariate/real_images.h5"
            try:
                with h5py.File(real_path, "r") as h5_file:
                    real_images = h5_file["samples"][()]

                fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

                if grid_size == 1:
                    axes = axes.reshape(1, 1)

                for i in range(grid_size * grid_size):
                    row, col = divmod(i, grid_size)
                    ax = axes[row, col]

                    img = np.transpose(real_images[i, :, :, :], (1, 2, 0))
                    ax.imshow(img, cmap=cmap)
                    ax.axis("off")

                fig.suptitle(
                    f"{dataset} - Real Images (Reference)", fontsize=18, y=0.95
                )
                plt.subplots_adjust(wspace=0.1, hspace=0.1)

                filename = f"{dataset.lower()}_real_reference.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Saved reference: {filename}")
                return  # Exit after finding first valid real images

            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing real images for {dataset}: {e}")
                continue

    print(f"Warning: Could not find real images for {dataset}")


def main():
    """Generate all individual plots."""
    print("Generating individual plots for each prior-function combination...")

    for dataset, config in DATASETS.items():
        print(f"\nProcessing dataset: {dataset}")
        plot_real_images_reference(dataset, config["grid_size"], config["cmap"])

        for prior in PRIORS:
            for function in FUNCTIONS:
                plot_prior_function_grid(
                    dataset, prior, function, config["grid_size"], config["cmap"]
                )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
