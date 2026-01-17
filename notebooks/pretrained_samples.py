#!/usr/bin/env python3

import argparse
import os

import h5py
import numpy as np
import torch
from diffusers import DDPMPipeline
from PIL import Image
from tqdm import tqdm

MODEL_CONFIGS = {
    "cifar10": ("google/ddpm-cifar10-32", 32, None),
    "celeba": ("google/ddpm-celebahq-256", 256, 64),
}


def generate_samples(pipeline, num_samples, batch_size, target_size=None):
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc="Generating"):
        current_batch = min(batch_size, num_samples - generated)
        if current_batch <= 0:
            break

        with torch.no_grad():
            if target_size:
                images = pipeline(batch_size=current_batch, output_type="pil").images
                batch_np = np.stack([
                    np.array(img.resize((target_size, target_size), Image.LANCZOS)).astype(np.float32) / 255.0
                    for img in images
                ], axis=-1)
            else:
                images = pipeline(batch_size=current_batch, output_type="np").images
                batch_np = np.transpose(images, (1, 2, 3, 0))

        samples.append(batch_np)
        generated += current_batch

    return np.concatenate(samples, axis=-1)[:, :, :, :num_samples].astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"])
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="logs/Pretrained")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_id, native_res, target_size = MODEL_CONFIGS[args.dataset]
    batch_size = min(args.batch_size, 16) if native_res >= 256 else args.batch_size

    print(f"Loading {model_id}...")
    pipeline = DDPMPipeline.from_pretrained(model_id).to(device)

    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples(pipeline, args.num_samples, batch_size, target_size)

    output_path = f"{args.output_dir}/{args.dataset.upper()}/DDPM/generated_images.h5"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("samples", data=samples, compression="gzip")

    print(f"Saved to {output_path}")
    print(f"Shape: {samples.shape}, Range: [{samples.min():.3f}, {samples.max():.3f}]")


if __name__ == "__main__":
    main()
