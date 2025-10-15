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
    # "SVHN_vanilla_mixture": {
    #     "dataset": "SVHN",
    #     "method_type": "Vanilla",
    #     "sampler": "ULA",
    #     "model_type": "mixture",
    #     "grid_size": 10,
    #     "cmap": None,
    #     "epochs": list(range(0, 41, 2)),
    #     "samples_per_frame": 1,
    #     "filename": "svhn_vanilla_ula_mixture_evolution.gif",
    # },
    "SVHN_thermo_mixture": {
        "dataset": "SVHN",
        "method_type": "Thermodynamic",
        "sampler": "ULA",
        "model_type": "mixture",
        "grid_size": 10,
        "cmap": None,
        "epochs": list(range(0, 41, 2)),
        "samples_per_frame": 1,
        "filename": "svhn_thermo_ula_mixture_evolution.gif",
    },
    # "CELEBA_vanilla_mixture": {
    #     "dataset": "CELEBA",
    #     "method_type": "Vanilla",
    #     "sampler": "ULA",
    #     "model_type": "mixture",
    #     "grid_size": 10,
    #     "cmap": None,
    #     "epochs": list(range(0, 41, 2)),
    #     "samples_per_frame": 1,
    #     "filename": "CELEBA_vanilla_ula_mixture_evolution.gif",
    # },
    "CELEBA_thermo_mixture": {
        "dataset": "CELEBA",
        "method_type": "Thermodynamic",
        "sampler": "ULA",
        "model_type": "mixture",
        "grid_size": 10,
        "cmap": None,
        "epochs": list(range(0, 41, 2)),
        "samples_per_frame": 1,
        "filename": "CELEBA_thermo_ula_mixture_evolution.gif",
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
    elif len(axes.shape) == 1:
        axes = axes.reshape(-1, 1)

    start_idx = (sample_subset_idx * grid_size * grid_size) % total_samples
    end_idx = min(start_idx + grid_size * grid_size, total_samples)

    for i in range(grid_size * grid_size):
        row, col = divmod(i, grid_size)
        ax = axes[row, col]

        sample_idx = start_idx + i
        if sample_idx < total_samples:
            img = np.transpose(generated_images[sample_idx, :, :, :], (1, 2, 0))

            if cmap is None:
                if img.max() > 1.0:
                    img = img / 255.0
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
        else:
            if cmap is None:
                ax.imshow(np.zeros((32, 32, 3)))
            else:
                ax.imshow(np.zeros((32, 32)), cmap="gray")
        ax.axis("off")

    method_label = (
        f"{config['method_type']} - {config['sampler']} - {config['model_type']}"
    )

    epoch_idx = config["epochs"].index(epoch)
    total_frames = len(config["epochs"]) * config["samples_per_frame"]
    current_frame = epoch_idx * config["samples_per_frame"] + sample_subset_idx
    total_progress = (current_frame + 1) / total_frames
    progress_filled = int(total_progress * 20)
    progress_bar = "=" * progress_filled + "." * (20 - progress_filled)

    title = f"{config['dataset']} - {method_label}\nTraining: [{progress_bar}] ({current_frame + 1}/{total_frames})"
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
        f"Creating GIF for {config['dataset']} - {config['method_type']} - {config['sampler']} - {config['model_type']}"
    )

    frames = []
    valid_epochs = []
    samples_per_frame = config["samples_per_frame"]

    for epoch in config["epochs"]:
        file_path = f"logs/{config['method_type']}/{config['dataset']}/{config['sampler']}/{config['model_type']}/generated_images_epoch_{epoch}.h5"

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
            print(f"  Epoch {epoch}: ✗ (file not found at {file_path})")
            continue
        except Exception as e:
            print(f"  Epoch {epoch}: ✗ (error: {e})")
            continue

    if not frames:
        print(
            f"  No valid frames found for {config['dataset']} - {config['method_type']} - {config['sampler']} - {config['model_type']}"
        )
        return

    output_path = os.path.join(output_dir, config["filename"])

    imageio.mimsave(
        output_path,
        frames,
        duration=300,
    )

    total_frames = len(frames)
    print(f"  Saved GIF: {config['filename']} ({total_frames} frames)")


def main():
    print("Creating sample evolution GIFs for ULA methods...")
    print(f"Output directory: {output_dir}")

    for config_name, config in GIF_CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Plotting: {config_name}")
        print(f"{'='*60}")

        create_evolution_gif(config)


if __name__ == "__main__":
    main()
