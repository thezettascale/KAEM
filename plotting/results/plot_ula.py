import os
import tempfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
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
    "SVHN": {"grid_size": 12, "cmap": None},
    "CELEBA": {"grid_size": 12, "cmap": None},
}

METHOD_CONFIGS = {
    "vanilla_ula_mixture": {
        "method_type": "Vanilla",
        "sampler": "ULA",
        "model_type": "mixture",
    },
    "thermo_ula_mixture": {
        "method_type": "Thermodynamic",
        "sampler": "ULA",
        "model_type": "mixture",
    },
}

output_dir = "figures/results/individual_plots"
os.makedirs(output_dir, exist_ok=True)


def select_best_samples_fast(generated_images, num_samples):
    """Select the best samples based on bootstrap metrics."""
    if generated_images.shape[0] <= num_samples:
        return np.arange(generated_images.shape[0])

    quality_scores = []

    for i in range(generated_images.shape[0]):
        img = np.transpose(generated_images[i, :, :, :], (1, 2, 0))

        if img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0, 1)

        # Grayscale
        if img.shape[2] == 3:
            gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            gray = img[:, :, 0]

        # 1. High variance better
        variance = np.var(gray)

        # 2. Edge content by Sobel filter (higher is better)
        sobel_x = ndimage.sobel(gray, axis=0)
        sobel_y = ndimage.sobel(gray, axis=1)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_content = np.mean(edge_magnitude)

        # 3. Avoid images with too little dynamic range
        dynamic_range = np.max(gray) - np.min(gray)

        quality_score = variance * 0.2 + edge_content * 0.0 + dynamic_range * 0.7
        quality_scores.append(quality_score)

    best_indices = np.argsort(quality_scores)[-num_samples:][
        ::-1
    ]  # Reverse to get highest first
    return best_indices


def plot_generated_images_grid(dataset, method_config, grid_size, cmap):
    """Generate and save a single plot for generated images from ULA method."""

    gen_path = f"logs/{method_config['method_type']}/{dataset}/{method_config['sampler']}/{method_config['model_type']}/generated_images.h5"

    try:
        with h5py.File(gen_path, "r") as h5_file:
            generated_images = h5_file["samples"][()]

        best_indices = select_best_samples_fast(generated_images, grid_size * grid_size)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

        if grid_size == 1:
            axes = axes.reshape(1, 1)
        elif len(axes.shape) == 1:
            axes = axes.reshape(-1, 1)

        for i in range(grid_size * grid_size):
            row, col = divmod(i, grid_size)
            ax = axes[row, col]

            if i < len(best_indices):
                img = np.transpose(
                    generated_images[best_indices[i], :, :, :], (1, 2, 0)
                )

                if cmap is None:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap=cmap)
            else:
                if cmap is None:
                    ax.imshow(np.zeros((32, 32, 3)))
                else:
                    ax.imshow(np.zeros((32, 32)), cmap="gray")
            ax.axis("off")

        method_label = f"{method_config['method_type']} - {method_config['sampler']} - {method_config['model_type']}"
        fig.suptitle(f"{dataset} - {method_label}", fontsize=18, y=0.95)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        filename = f"{dataset.lower()}_{method_config['method_type'].lower()}_{method_config['sampler'].lower()}_{method_config['model_type']}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    except FileNotFoundError:
        print(f"Warning: Could not find data for {dataset} - {method_label}")
    except Exception as e:
        print(f"Error processing {dataset} - {method_label}: {e}")


def plot_real_images_reference(dataset, grid_size, cmap):
    """Generate a reference plot with real images for each dataset."""

    for config_name, method_config in METHOD_CONFIGS.items():
        real_path = f"logs/{method_config['method_type']}/{dataset}/{method_config['sampler']}/{method_config['model_type']}/real_images.h5"
        try:
            with h5py.File(real_path, "r") as h5_file:
                real_images = h5_file["samples"][()]

            fig, axes = plt.subplots(grid_size, grid_size, figsize=(6, 6))

            if grid_size == 1:
                axes = axes.reshape(1, 1)
            elif len(axes.shape) == 1:
                axes = axes.reshape(-1, 1)

            for i in range(grid_size * grid_size):
                row, col = divmod(i, grid_size)
                ax = axes[row, col]

                if i < real_images.shape[0]:
                    img = np.transpose(real_images[i, :, :, :], (1, 2, 0))

                    if cmap is None:
                        ax.imshow(img)
                    else:
                        ax.imshow(img, cmap=cmap)
                else:
                    if cmap is None:
                        ax.imshow(np.zeros((32, 32, 3)))
                    else:
                        ax.imshow(np.zeros((32, 32)), cmap="gray")
                ax.axis("off")

            fig.suptitle(f"{dataset} - Real Images (Reference)", fontsize=18, y=0.95)
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            filename = f"{dataset.lower()}_real_reference.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Saved reference: {filename}")
            return

        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Error processing real images for {dataset}: {e}")
            continue

    print(f"Warning: Could not find real images for {dataset}")


def main():
    """Generate all individual plots for ULA methods."""
    print("Generating individual plots for ULA methods...")

    for dataset, config in DATASETS.items():
        print(f"\nProcessing dataset: {dataset}")

        plot_real_images_reference(dataset, config["grid_size"], config["cmap"])

        for config_name, method_config in METHOD_CONFIGS.items():
            plot_generated_images_grid(
                dataset, method_config, config["grid_size"], config["cmap"]
            )

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
