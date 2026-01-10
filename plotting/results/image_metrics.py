import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch_fidelity import calculate_metrics


def get_num_gpus():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def process_on_device(args):
    gen_file_path, real_file_path, device_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    metrics = calculate_infinity_metrics(gen_file_path, real_file_path)

    log_dir = os.path.join(os.path.dirname(gen_file_path), "metrics")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "image_metrics.json")
    with open(log_file, "w") as f:
        json.dump(metrics, f, indent=4)

    return gen_file_path, metrics


def load_images(file_path):
    with h5py.File(file_path, "r") as f:
        images = np.array(f["samples"])
    return torch.tensor(images)


def save_images_to_directory(images, directory):
    os.makedirs(directory, exist_ok=True)
    for i, img_tensor in enumerate(images):
        img = img_tensor.permute(1, 2, 0).numpy() * 255
        img = img.astype(np.uint8)
        Image.fromarray(img).save(os.path.join(directory, f"image_{i}.png"))


def calculate_infinity_metrics(
    gen_file_path,
    real_file_path,
    batch_sizes=[1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 3000],
):
    real_images = load_images(real_file_path)
    gen_images = load_images(gen_file_path)

    with tempfile.TemporaryDirectory() as real_images_dir:
        save_images_to_directory(real_images, real_images_dir)
        fids = []
        kids = []

        for batch_size in batch_sizes:
            indices = np.random.choice(len(gen_images), batch_size, replace=False)
            with tempfile.TemporaryDirectory() as gen_images_dir:
                save_images_to_directory(gen_images[indices], gen_images_dir)
                metrics = calculate_metrics(
                    input1=real_images_dir,
                    input2=gen_images_dir,
                    fid=True,
                    kid=True,
                )
                fids.append(metrics["frechet_inception_distance"])
                kids.append(metrics["kernel_inception_distance_mean"])

        inverse_batch_sizes = 1 / np.array(batch_sizes).reshape(-1, 1)
        fid_reg = LinearRegression().fit(
            inverse_batch_sizes, np.array(fids).reshape(-1, 1)
        )
        kid_reg = LinearRegression().fit(
            inverse_batch_sizes, np.array(kids).reshape(-1, 1)
        )

        fid_infinity = fid_reg.predict(np.array([[0]]))[0, 0]
        kid_infinity = kid_reg.predict(np.array([[0]]))[0, 0]

        return {
            "fid_infinity": float(fid_infinity),
            "kid_infinity": float(kid_infinity),
            "fid_values": [float(f) for f in fids],
            "kid_values": [float(k) for k in kids],
            "batch_sizes": batch_sizes,
        }


def run_distributed(file_paths, num_workers=None):
    num_gpus = get_num_gpus()
    if num_workers is None:
        num_workers = max(1, num_gpus)

    print(f"Running with {num_workers} worker(s) across {num_gpus} GPU(s)")

    tasks = [(gen, real, i % num_workers) for i, (gen, real) in enumerate(file_paths)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_on_device, task): task for task in tasks}

        for future in as_completed(futures):
            gen_file_path, metrics = future.result()
            print(f"Processed {gen_file_path}")
            print(f"FID  : {metrics['fid_infinity']:.2f}")
            print(f"KID  : {metrics['kid_infinity']:.2f}")
            print("---")


if __name__ == "__main__":
    file_paths = [
        # KAEM - SVHN
        (
            "logs/Vanilla/SVHN/ULA/mixture/generated_images.h5",
            "logs/Vanilla/SVHN/ULA/mixture/real_images.h5",
        ),
        (
            "logs/Thermodynamic/SVHN/ULA/mixture/generated_images.h5",
            "logs/Thermodynamic/SVHN/ULA/mixture/real_images.h5",
        ),
        # KAEM - CELEBA
        (
            "logs/Vanilla/CELEBA/ULA/mixture/generated_images.h5",
            "logs/Vanilla/CELEBA/ULA/mixture/real_images.h5",
        ),
        (
            "logs/Thermodynamic/CELEBA/ULA/mixture/generated_images.h5",
            "logs/Thermodynamic/CELEBA/ULA/mixture/real_images.h5",
        ),
        # KAEM - CIFAR10
        (
            "logs/Vanilla/CIFAR10/ULA/mixture/generated_images.h5",
            "logs/Vanilla/CIFAR10/ULA/mixture/real_images.h5",
        ),
        (
            "logs/Thermodynamic/CIFAR10/ULA/mixture/generated_images.h5",
            "logs/Thermodynamic/CIFAR10/ULA/mixture/real_images.h5",
        ),
        # Baseline - CIFAR10
        (
            "logs/Baseline/CIFAR10/VAE/generated_images.h5",
            "logs/Baseline/CIFAR10/VAE/real_images.h5",
        ),
        (
            "logs/Baseline/CIFAR10/GAN/generated_images.h5",
            "logs/Baseline/CIFAR10/GAN/real_images.h5",
        ),
        (
            "logs/Baseline/CIFAR10/DDPM/generated_images.h5",
            "logs/Baseline/CIFAR10/DDPM/real_images.h5",
        ),
        # Baseline - CELEBA
        (
            "logs/Baseline/CELEBA/VAE/generated_images.h5",
            "logs/Baseline/CELEBA/VAE/real_images.h5",
        ),
        (
            "logs/Baseline/CELEBA/GAN/generated_images.h5",
            "logs/Baseline/CELEBA/GAN/real_images.h5",
        ),
        (
            "logs/Baseline/CELEBA/DDPM/generated_images.h5",
            "logs/Baseline/CELEBA/DDPM/real_images.h5",
        ),
        # Baseline - SVHN
        (
            "logs/Baseline/SVHN/VAE/generated_images.h5",
            "logs/Baseline/SVHN/VAE/real_images.h5",
        ),
        (
            "logs/Baseline/SVHN/GAN/generated_images.h5",
            "logs/Baseline/SVHN/GAN/real_images.h5",
        ),
        (
            "logs/Baseline/SVHN/DDPM/generated_images.h5",
            "logs/Baseline/SVHN/DDPM/real_images.h5",
        ),
    ]

    run_distributed(file_paths)
