<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/brand/banner-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="docs/assets/brand/banner-light.svg">
  <img alt="Latent: local image generation and editing on your own GPU" src="docs/assets/brand/banner-dark.svg" width="100%">
</picture>

**Write a prompt or drop in a photo, and watch the image develop.<br>Runs on your own GPU, on Windows and Linux. Nothing leaves your computer.**

[![Status: in development](https://img.shields.io/badge/status-in%20development-ffaa2c?style=flat-square)](docs/PLAN.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-3987e5?style=flat-square)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux-555?style=flat-square)](#install)
[![Built with Tauri 2 and Rust](https://img.shields.io/badge/built%20with-Tauri%202%20%2B%20Rust-24C8DB?style=flat-square&logo=tauri&logoColor=white)](https://tauri.app)

[Features](#features) · [Models](#models) · [How it works](#how-it-works) · [Install](#install) · [Responsible use](#responsible-use) · [Privacy](#privacy)

</div>

> [!NOTE]
> Latent is in early development and there is nothing to download yet. This page describes the first release; the design and the plan are in [docs/PLAN.md](docs/PLAN.md). From the maker of [Mimic](https://github.com/Mazon3442/Mimic), which does the same for voices.

## Features

| Feature | What it does |
| --- | --- |
| **Watch it develop** | Each frame appears in its tile as it's made, a step at a time, so a bad seed shows in the first seconds and you can cancel it. The model stays loaded. |
| **Edit any photo** | Drop, paste or import a picture and describe the change. The canvas shows the result next to the original. |
| **Use as source** | Any frame becomes the input for the next edit with one key (E). Make, pick, refine, without downloading and uploading. |
| **The right size for each model** | Choose an aspect ratio, not pixel counts. Latent picks the model's native resolution, so results don't come out stretched or cropped. |
| **Models that fit your PC** | Latent reads your GPU and memory and offers each model in a variant that fits, quantized or offloaded when needed. The bar at the top shows real VRAM use, and hovering a model shows its footprint before you load it. |
| **One model in memory** | Switching models unloads the old one first, and idle models unload after a set time, so the GPU is free for everything else. |
| **Nothing saved behind your back** | Frames stay in a temporary cache until you press Save (Ctrl+S) or Save as (Ctrl+Shift+S). PNG, WebP or JPEG. |
| **Every frame keeps its recipe** | Model, prompt, size, seed and steps go into the saved file. Reuse a recipe (R), make more like it (V), or open a saved image to load its recipe. |
| **Tools** | Upscale 2x or 4x and remove backgrounds on any frame or photo, on your own hardware. |

## Models

| Model | Does | Memory | License |
| --- | --- | --- | --- |
| **Z-Image Turbo** | generate, in 8 steps | 12–16 GB VRAM¹ | Apache 2.0 |
| **FLUX.2 [klein] 4B** | generate and edit, several reference images | 10–13 GB VRAM¹ | Apache 2.0 |
| **Qwen-Image-Edit** | edit, strong on text and objects | 16 GB VRAM, quantized¹ | Apache 2.0 |
| **SDXL 1.0** | generate, inpaint; runs on smaller GPUs | about 8 GB VRAM¹ | CreativeML OpenRAIL++-M |
| **Real-ESRGAN** | upscale | under 1 GB | BSD-3-Clause |
| **BiRefNet** | remove background | about 2 GB | MIT |

<sub>¹ Published or estimated figures, not yet measured with Latent. They will be replaced with measurements from an RTX 5070 Ti and a 16-thread CPU before the first release. No model weights are included in Latent; each is downloaded on request and keeps its own license, which applies to what you make with it.</sub>

## How it works

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/charts/architecture-dark.svg">
  <img alt="Diagram: the interface talks to a Rust core through Tauri commands and events; the core runs a diffusion worker and a tools worker over JSON lines; models run on an NVIDIA GPU, an AMD GPU (Linux), an Intel Arc GPU or the CPU." src="docs/assets/charts/architecture-light.svg" width="760">
</picture>
</div>

The interface is plain HTML, CSS and JavaScript in the system web view. A small Rust core starts the models in a separate Python process built on [diffusers](https://github.com/huggingface/diffusers), and puts it in a job (Windows) or process group (Linux) that ends with Latent, so even a crash can't leave a model holding your VRAM. The core keeps one model in memory at a time and stops a run between steps when you cancel.

## Install

Not yet. The first release will ship like Mimic's:

| Platform | Download |
| --- | --- |
| **Windows 10/11** | `Latent_<version>_x64-setup.exe` from Releases |
| **Debian / Ubuntu** | `.deb` from Releases |
| **Fedora** | `.rpm` from Releases |
| **Any Linux** | `.AppImage` from Releases |

Latent will also need **[uv](https://docs.astral.sh/uv/)** to set up the models (`winget install astral-sh.uv` on Windows), and enough disk for the models you choose: 8 to 30 GB each.

<details>
<summary><b>Repository layout</b></summary>

```text
app/ui/                 interface (HTML, CSS, JS; fonts bundled)
app/src-tauri/          Rust core: model manager, installer, hardware, sources and frames
engines/                Python workers: diffusion models and tools
design/                 the clickable design mockup, and brand.py, which draws the logo and charts
docs/                   the plan and README assets
```

</details>

## Responsible use

Latent can edit photos of real people, so it can be misused. Edit your own photos, or someone else's only with their permission. Making sexual or degrading images of real people, deceiving people with fake photos, or impersonating someone is illegal in many places and is not what Latent is for. You're responsible for what you make with it.

- Images you save are marked as AI-generated in the file's metadata. When you share them, say so.
- Each model's license applies to what you make with it; check it before commercial use.

## Privacy

Latent has no accounts and no telemetry. Your photos, prompts and images stay on your computer. It will connect to the internet only to check GitHub for a new version when it starts and to download the models you ask for (PyPI, the PyTorch index and Hugging Face). Those services see the usual request details, such as your IP address.

## License

Latent's code is released under the [MIT License](LICENSE). No model weights are included. Each model has its own license, so check it before any commercial use.

<div align="center">
<br>
<img src="docs/assets/brand/mark-animated.svg" alt="" width="40">
</div>
