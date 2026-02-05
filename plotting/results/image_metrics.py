import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression
from torch_fidelity import calculate_metrics
import argparse


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
    batch_sizes=[2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
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
        fid_y = np.array(fids).reshape(-1, 1)
        kid_y = np.array(kids).reshape(-1, 1)

        fid_reg = LinearRegression().fit(inverse_batch_sizes, fid_y)
        kid_reg = LinearRegression().fit(inverse_batch_sizes, kid_y)

        fid_infinity = fid_reg.predict(np.array([[0]]))[0, 0]
        kid_infinity = kid_reg.predict(np.array([[0]]))[0, 0]

        # Compute R² scores
        fid_r2 = fid_reg.score(inverse_batch_sizes, fid_y)
        kid_r2 = kid_reg.score(inverse_batch_sizes, kid_y)

        # Compute standard error of intercept for uncertainty estimation
        n = len(batch_sizes)
        x = inverse_batch_sizes.flatten()
        x_mean = np.mean(x)
        ss_x = np.sum((x - x_mean) ** 2)

        # FID standard error
        fid_pred = fid_reg.predict(inverse_batch_sizes).flatten()
        fid_residuals = fid_y.flatten() - fid_pred
        fid_mse = np.sum(fid_residuals**2) / (n - 2)
        fid_se_intercept = np.sqrt(fid_mse * (1 / n + x_mean**2 / ss_x))

        # KID standard error
        kid_pred = kid_reg.predict(inverse_batch_sizes).flatten()
        kid_residuals = kid_y.flatten() - kid_pred
        kid_mse = np.sum(kid_residuals**2) / (n - 2)
        kid_se_intercept = np.sqrt(kid_mse * (1 / n + x_mean**2 / ss_x))

        return {
            "fid_infinity": float(fid_infinity),
            "fid_std_error": float(fid_se_intercept),
            "fid_r2": float(fid_r2),
            "kid_infinity": float(kid_infinity),
            "kid_std_error": float(kid_se_intercept),
            "kid_r2": float(kid_r2),
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
            fid = metrics["fid_infinity"]
            fid_se = metrics["fid_std_error"]
            fid_r2 = metrics["fid_r2"]
            kid = metrics["kid_infinity"]
            kid_se = metrics["kid_std_error"]
            kid_r2 = metrics["kid_r2"]
            print(f"Processed {gen_file_path}")
            print(f"FID  : {fid:.2f} ± {fid_se:.2f} (R²={fid_r2:.3f})")
            print(f"KID  : {kid:.4f} ± {kid_se:.4f} (R²={kid_r2:.3f})")
            print("---")


def get_real_samples_path(dataset: str) -> str:
    return f"logs/RealSamples/{dataset}/real_images.h5"


def discover_generated_samples(logs_dir: str = "logs") -> list[tuple[str, str]]:
    file_paths = []
    datasets = ["CIFAR10", "CELEBA", "SVHN"]

    for dataset in datasets:
        real_path = get_real_samples_path(dataset)
        if not os.path.exists(real_path):
            print(f"Warning: Real samples not found for {dataset} at {real_path}")
            continue

        # KAEM models - Vanilla
        for train_type in ["ULA", "importance", "amortized"]:
            for prior_type in ["mixture", "univariate"]:
                gen_path = f"{logs_dir}/Vanilla/{dataset}/{train_type}/{prior_type}/generated_images.h5"
                if os.path.exists(gen_path):
                    file_paths.append((gen_path, real_path))

        # KAEM models - Thermodynamic
        for train_type in ["ULA", "importance", "amortized"]:
            for prior_type in ["mixture", "univariate"]:
                gen_path = f"{logs_dir}/Thermodynamic/{dataset}/{train_type}/{prior_type}/generated_images.h5"
                if os.path.exists(gen_path):
                    file_paths.append((gen_path, real_path))

        # Baseline models
        for model in ["VAE", "GAN", "DDPM", "PANG"]:
            gen_path = f"{logs_dir}/Baseline/{dataset}/{model}/generated_images.h5"
            if os.path.exists(gen_path):
                file_paths.append((gen_path, real_path))

        # Pretrained models
        for model in ["DDPM"]:
            gen_path = f"{logs_dir}/Pretrained/{dataset}/{model}/generated_images.h5"
            if os.path.exists(gen_path):
                file_paths.append((gen_path, real_path))

    return file_paths


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Compute FID/KID metrics for generated samples"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Auto-discover all generated samples"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Filter by dataset (CIFAR10, CELEBA, SVHN)",
    )
    args = parser.parse_args()

    if args.auto:
        file_paths = discover_generated_samples()
        if args.dataset:
            file_paths = [(g, r) for g, r in file_paths if args.dataset.upper() in g]
    else:
        # Manual list for specific runs
        file_paths = [
            # # KAEM - CIFAR10
            # (
            #     "logs/Vanilla/CIFAR10/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("CIFAR10"),
            # ),
            # (
            #     "logs/Thermodynamic/CIFAR10/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("CIFAR10"),
            # ),
            # KAEM - CELEBA
            # (
            #     "logs/Vanilla/CELEBA/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("CELEBA"),
            # ),
            # (
            #     "logs/Thermodynamic/CELEBA/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("CELEBA"),
            # ),
            # # KAEM - SVHN
            # (
            #     "logs/Vanilla/SVHN/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("SVHN"),
            # ),
            # (
            #     "logs/Thermodynamic/SVHN/ULA/mixture/generated_images.h5",
            #     get_real_samples_path("SVHN"),
            # ),
            # # Baselines - VAE
            # (
            #     "logs/Baseline/CIFAR10/VAE/generated_images.h5",
            #     get_real_samples_path("CIFAR10"),
            # ),
            (
                "logs/Baseline/CELEBA/VAE/generated_images.h5",
                get_real_samples_path("CELEBA"),
            ),
            (
                "logs/Baseline/SVHN/VAE/generated_images.h5",
                get_real_samples_path("SVHN"),
            ),
            # (
            #     "logs/Baseline/CIFAR10/GAN/generated_images.h5",
            #     get_real_samples_path("CIFAR10"),
            # ),
            # # Pretrained - CIFAR10
            # (
            #     "logs/Pretrained/CIFAR10/DDPM/generated_images.h5",
            #     get_real_samples_path("CIFAR10"),
            # ),
        ]

    # Filter to only existing files
    file_paths = [
        (g, r) for g, r in file_paths if os.path.exists(g) and os.path.exists(r)
    ]

    if not file_paths:
        print("No generated samples found. Run training first or check paths.")
    else:
        print(f"Found {len(file_paths)} generated sample files to evaluate:")
        for g, r in file_paths:
            print(f"  {g}")
        print()
        run_distributed(file_paths)
