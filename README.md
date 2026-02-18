# AI Video Generator

This project includes a runnable Python script that generates short, high-quality AI videos using open-source models from Hugging Face.

## What it does

- **Text → Image** with SDXL (`stabilityai/sdxl-turbo` by default)
- **Image → Video segments** with Stable Video Diffusion (`stabilityai/stable-video-diffusion-img2vid-xt`)
- **Auto-stitches segments** so you can generate longer videos (up to **30 seconds**)
- Exports an MP4 file

## Setup (Windows Terminal)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in Windows Terminal and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Setup (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> First run downloads model weights (several GB).

## Usage

### 1) Generate from a text prompt (Windows Terminal)

```powershell
python .\ai_video_generator.py `
  --prompt "cinematic drone shot of a futuristic city at sunrise, ultra detailed" `
  --duration-seconds 30 `
  --fps 8 `
  --output .\outputs\city_30s.mp4
```

### 2) Generate from an existing image (Windows Terminal)

```powershell
python .\ai_video_generator.py `
  --input-image ".\media\Screenshot 2024-07-20 142410.png" `
  --duration-seconds 30 `
  --output .\outputs\from_image_30s.mp4
```

### 3) macOS/Linux example

```bash
python ai_video_generator.py \
  --prompt "cinematic drone shot of a futuristic city at sunrise, ultra detailed" \
  --duration-seconds 30 \
  --fps 8 \
  --output outputs/city_30s.mp4
```

## Key quality controls

- `--duration-seconds` (default: `5.0`, max: `30`)
- `--fps` (default: `8`): higher values look smoother but require more generation
- `--motion-bucket-id` (default: `127`): controls motion intensity
- `--steps` and `--guidance`: image quality controls for text-to-image stage
- `--width` and `--height`: output resolution

## Notes

- Best results are on an NVIDIA GPU with sufficient VRAM.
- On CPU generation can be very slow.
- If your environment requires authentication for gated models, run:

```bash
huggingface-cli login
```
