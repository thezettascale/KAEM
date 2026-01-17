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
    "cifar10": {"ddpm": ("google/ddpm-cifar10-32", 32, False)},
    "celeba": {"ddpm": ("google/ddpm-celebahq-256", 256, True)},
}

TARGET_RESOLUTIONS = {"cifar10": 32, "celeba": 64}


def load_pipeline(model_id: str, device: str) -> DDPMPipeline:
    print(f"Loading {model_id}...")
    pipeline = DDPMPipeline.from_pretrained(model_id)
    return pipeline.to(device)


def generate_samples(
    pipeline: DDPMPipeline,
    num_samples: int,
    batch_size: int,
    target_size: int | None = None,
    desc: str = "Generating",
) -> np.ndarray:
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    generated = 0

    for _ in tqdm(range(num_batches), desc=desc):
        current_batch = min(batch_size, num_samples - generated)
        if current_batch <= 0:
            break

        with torch.no_grad():
            if target_size is not None:
                images = pipeline(batch_size=current_batch, output_type="pil").images
                batch_np = np.stack(
                    [
                        np.array(
                            img.resize((target_size, target_size), Image.LANCZOS)
                        ).astype(np.float32)
                        / 255.0
                        for img in images
                    ],
                    axis=-1,
                )
            else:
                images = pipeline(batch_size=current_batch, output_type="np").images
                batch_np = np.transpose(images, (1, 2, 3, 0))

        samples.append(batch_np)
        generated += current_batch

    return np.concatenate(samples, axis=-1)[:, :, :, :num_samples].astype(np.float32)


def save_samples(samples: np.ndarray, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    with h5py.File(output_path, "w") as f:
        f.create_dataset("samples", data=samples, compression="gzip")

    print(f"Saved {samples.shape[-1]} samples to {output_path}")
    print(
        f"  Shape: {samples.shape}, Range: [{samples.min():.3f}, {samples.max():.3f}]"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ddpm", choices=["ddpm"])
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=["cifar10", "celeba"]
    )
    parser.add_argument("--num_samples", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="logs/Pretrained")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if (
        args.dataset not in MODEL_CONFIGS
        or args.model not in MODEL_CONFIGS[args.dataset]
    ):
        print(f"Model {args.model} not available for {args.dataset}")
        return

    model_id, native_res, needs_resize = MODEL_CONFIGS[args.dataset][args.model]
    target_size = TARGET_RESOLUTIONS[args.dataset] if needs_resize else None
    batch_size = min(args.batch_size, 16) if native_res >= 256 else args.batch_size

    pipeline = load_pipeline(model_id, args.device)
    samples = generate_samples(
        pipeline,
        args.num_samples,
        batch_size,
        target_size=target_size,
        desc=f"Generating {args.model.upper()} {args.dataset.upper()}",
    )

    output_path = f"{args.output_dir}/{args.dataset.upper()}/{args.model.upper()}/generated_images.h5"
    save_samples(samples, output_path)


if __name__ == "__main__":
    main()
