import os
from pathlib import Path

import h5py
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

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

GIF_CONFIGS = {
    "MNIST_ebm_fft": {
        "dataset": "MNIST",
        "prior": "ebm",
        "function": "FFT",
        "grid_size": 6,
        "cmap": "gray",
        "epochs": list(range(1, 11)),
        "samples_per_frame": 6,
        "filename": "mnist_ebm_fft_evolution.gif",
    },
    "FMNIST_gaussian_rbf": {
        "dataset": "FMNIST",
        "prior": "gaussian",
        "function": "RBF",
        "grid_size": 6,
        "cmap": "gray",
        "epochs": list(range(1, 11)),
        "samples_per_frame": 6,
        "filename": "fmnist_gaussian_rbf_evolution.gif",
    },
    "DARCY_FLOW_gaussian_fft": {
        "dataset": "DARCY_FLOW",
        "prior": "gaussian",
        "function": "FFT",
        "grid_size": 6,
        "cmap": "viridis",
        "epochs": [
            0,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
        ],  # Darcy uses 100-step increments
        "samples_per_frame": 6,
        "filename": "darcy_flow_gaussian_fft_evolution.gif",
    },
}

output_dir = "figures/results/gif_evolution"
os.makedirs(output_dir, exist_ok=True)


def create_sample_subset_frame(
    config, epoch, sample_subset_idx, generated_images, total_samples
):
    grid_size = config["grid_size"]
    cmap = config["cmap"]
    samples_per_frame = config["samples_per_frame"]

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    if grid_size == 1:
        axes = axes.reshape(1, 1)

    start_idx = (sample_subset_idx * grid_size * grid_size) % total_samples
    end_idx = min(start_idx + grid_size * grid_size, total_samples)

    for i in range(grid_size * grid_size):
        row, col = divmod(i, grid_size)
        ax = axes[row, col]

        sample_idx = start_idx + i
        if sample_idx < total_samples:
            img = np.transpose(generated_images[sample_idx, :, :, :], (1, 2, 0))
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(np.zeros_like(generated_images[0, :, :, :]), cmap="gray")
        ax.axis("off")

    prior_labels = {
        "uniform": r"$\mathcal{U}(\bm{z}; \; \bm{0}, \bm{1})$",
        "lognormal": r"$\text{Lognormal}(\bm{z}; \; \bm{0}, \bm{1})$",
        "gaussian": r"$\mathcal{N}(\bm{z}; \; \bm{0}, \bm{1})$",
        "ebm": "EBM",
    }

    epoch_idx = config["epochs"].index(epoch)
    total_frames = len(config["epochs"]) * config["samples_per_frame"]
    current_frame = epoch_idx * config["samples_per_frame"] + sample_subset_idx
    total_progress = (current_frame + 1) / total_frames
    progress_filled = int(total_progress * 20)
    progress_bar = "=" * progress_filled + "." * (20 - progress_filled)

    title = f"{config['dataset']} - {prior_labels[config['prior']]} - {config['function']}\nTraining: [{progress_bar}] ({current_frame + 1}/{total_frames})"
    fig.suptitle(title, fontsize=14, y=0.95)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(-1, 4)[:, :3].flatten()
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return img_array


def create_evolution_gif(config):
    print(
        f"Creating GIF for {config['dataset']} - {config['prior']} - {config['function']}"
    )

    frames = []
    valid_epochs = []
    samples_per_frame = config["samples_per_frame"]

    for epoch in config["epochs"]:
        file_path = f"logs/Vanilla/{config['dataset']}/importance/{config['prior']}_{config['function']}/univariate/generated_images_epoch_{epoch}.h5"

        try:
            with h5py.File(file_path, "r") as h5_file:
                generated_images = h5_file["samples"][()]
                total_samples = generated_images.shape[0]

                print(f"  Epoch {epoch}: ✓ ({total_samples} samples)")

                for subset_idx in range(samples_per_frame):
                    frame = create_sample_subset_frame(
                        config, epoch, subset_idx, generated_images, total_samples
                    )
                    frames.append(frame)

                valid_epochs.append(epoch)

        except FileNotFoundError:
            print(f"  Epoch {epoch}: ✗ (file not found)")
            continue
        except Exception as e:
            print(f"  Epoch {epoch}: ✗ (error: {e})")
            continue

    if not frames:
        print(
            f"  No valid frames found for {config['dataset']} - {config['prior']} - {config['function']}"
        )
        return

    output_path = os.path.join(output_dir, config["filename"])

    imageio.mimsave(
        output_path,
        frames,
        duration=150,
    )

    total_frames = len(frames)
    print(f"  Saved GIF: {config['filename']}")


def main():
    print("Creating sample evolution GIFs...")
    print(f"Output directory: {output_dir}")

    for config_name, config in GIF_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Plotting: {config_name}")
        print(f"{'='*60}")

        create_evolution_gif(config)


if __name__ == "__main__":
    main()
