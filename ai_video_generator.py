#!/usr/bin/env python3
"""Generate high-quality videos with open-source diffusion models.

Pipeline:
1) text -> keyframe image (SDXL) OR use an input image
2) keyframe image -> short video segment (Stable Video Diffusion)
3) stitch segments to reach longer durations (up to 30 seconds)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from PIL import Image


DEFAULT_IMAGE_MODEL = "stabilityai/sdxl-turbo"
DEFAULT_VIDEO_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
MAX_DURATION_SECONDS = 30
MAX_FRAMES_PER_SVD_CALL = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AI videos from a prompt or image.")
    parser.add_argument("--prompt", type=str, help="Prompt to generate a starting image.")
    parser.add_argument(
        "--input-image",
        type=str,
        help="Path to a starting image. If set, --prompt is optional.",
    )
    parser.add_argument("--output", type=str, default="outputs/generated.mp4", help="Output video path.")
    parser.add_argument(
        "--image-model",
        type=str,
        default=DEFAULT_IMAGE_MODEL,
        help="Hugging Face model id for text-to-image.",
    )
    parser.add_argument(
        "--video-model",
        type=str,
        default=DEFAULT_VIDEO_MODEL,
        help="Hugging Face model id for image-to-video.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--width", type=int, default=1024, help="Image width.")
    parser.add_argument("--height", type=int, default=576, help="Image height.")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps for text-to-image.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale for text-to-image.")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for output MP4.")
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=5.0,
        help=f"Target duration in seconds (max {MAX_DURATION_SECONDS}).",
    )
    parser.add_argument(
        "--motion-bucket-id",
        type=int,
        default=127,
        help="Controls motion amount: larger values usually increase movement.",
    )
    parser.add_argument(
        "--noise-aug-strength",
        type=float,
        default=0.02,
        help="How much noise is added to each segment start frame.",
    )
    return parser.parse_args()


def pick_dtype(device: str) -> torch.dtype:
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_or_make_start_image(
    *,
    prompt: str | None,
    input_image: str | None,
    image_model: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    width: int,
    height: int,
    steps: int,
    guidance: float,
) -> Image.Image:
    if input_image:
        image = Image.open(input_image).convert("RGB")
        return image.resize((width, height), Image.Resampling.LANCZOS)

    if not prompt:
        raise ValueError("Provide --prompt or --input-image.")

    pipe = AutoPipelineForText2Image.from_pretrained(image_model, torch_dtype=dtype)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    return image


def _svd_segment(
    *,
    pipe: StableVideoDiffusionPipeline,
    start_image: Image.Image,
    device: str,
    seed: int,
    frames: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
) -> list[Image.Image]:
    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        start_image,
        decode_chunk_size=8,
        generator=generator,
        num_frames=frames,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
    )
    return result.frames[0]


def generate_video_frames(
    *,
    start_image: Image.Image,
    video_model: str,
    device: str,
    dtype: torch.dtype,
    seed: int,
    total_frames: int,
    motion_bucket_id: int,
    noise_aug_strength: float,
) -> list[np.ndarray]:
    if total_frames <= 0:
        raise ValueError("Total frame count must be positive.")

    variant = "fp16" if dtype == torch.float16 else None
    pipe = StableVideoDiffusionPipeline.from_pretrained(video_model, torch_dtype=dtype, variant=variant)
    pipe = pipe.to(device)

    all_frames: list[np.ndarray] = []
    current_image = start_image
    frames_left = total_frames
    segment_index = 0

    while frames_left > 0:
        segment_frames = min(MAX_FRAMES_PER_SVD_CALL, frames_left)
        seed_for_segment = seed + segment_index
        pil_frames = _svd_segment(
            pipe=pipe,
            start_image=current_image,
            device=device,
            seed=seed_for_segment,
            frames=segment_frames,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
        )

        if segment_index > 0 and pil_frames:
            pil_frames = pil_frames[1:]

        np_frames = [np.asarray(frame) for frame in pil_frames]
        all_frames.extend(np_frames)

        if pil_frames:
            current_image = pil_frames[-1]

        frames_left = total_frames - len(all_frames)
        segment_index += 1

    return all_frames[:total_frames]


def save_video(frames: list[np.ndarray], output_path: str, fps: int) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=9) as writer:
        for frame in frames:
            writer.append_data(frame)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()

    if args.duration_seconds <= 0:
        raise ValueError("--duration-seconds must be > 0.")
    if args.duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError(f"--duration-seconds must be <= {MAX_DURATION_SECONDS}.")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0.")

    total_frames = int(round(args.duration_seconds * args.fps))

    device = detect_device()
    dtype = pick_dtype(device)

    print(f"Using device: {device} ({dtype})")
    print(f"Target duration: {args.duration_seconds:.2f}s @ {args.fps} fps ({total_frames} frames)")

    start_image = load_or_make_start_image(
        prompt=args.prompt,
        input_image=args.input_image,
        image_model=args.image_model,
        device=device,
        dtype=dtype,
        seed=args.seed,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance=args.guidance,
    )

    frames = generate_video_frames(
        start_image=start_image,
        video_model=args.video_model,
        device=device,
        dtype=dtype,
        seed=args.seed,
        total_frames=total_frames,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
    )

    save_video(frames, args.output, args.fps)
    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
