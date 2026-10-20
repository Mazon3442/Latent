"""Diffusion worker for Latent. Run with diffusion-env python (-X utf8). Speaks JSON lines on stdin/stdout.

Start args: <family> <model folder> <offload: none | model>
Requests, one at a time (docs/PLAN.md section 5):
  {"op": "generate", "id", "prompt", "negative"?, "sources": [paths], "width", "height", "seed",
   "options": {"steps"?, "guidance"?}, "output", "recipe"}
  {"op": "cancel", "id"}      may arrive while a generate runs; stops it at the next step, model stays loaded
Events: ready, progress, done, cancelled, error.
"""
import json
import queue
import sys
import threading
import time

real_out = sys.stdout
sys.stdout = sys.stderr  # library prints must not corrupt the protocol stream
sys.stdin.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from device import pick

FAMILY = sys.argv[1]
FOLDER = sys.argv[2]
OFFLOAD = sys.argv[3] if len(sys.argv) > 3 else "none"
GIB = 1024 ** 3

_out = threading.Lock()


def send(**message):
    with _out:
        real_out.write(json.dumps(message) + "\n")
        real_out.flush()


class Cancelled(Exception):
    pass


def load(device):
    dtype = torch.bfloat16 if device != "cpu" else torch.float32  # bfloat16 is slow or unsupported on most CPUs
    if FAMILY == "zimage":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(FOLDER, torch_dtype=dtype)
    else:
        raise ValueError(f"unknown model family {FAMILY}")
    if device != "cpu" and OFFLOAD == "model":
        # Each component moves to the GPU only while it runs: the text encoder, then the transformer, then the VAE.
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def fit_preview(vae, device):
    """Least-squares map from model-space latent channels to RGB, fitted on this VAE.

    Encodes smooth random colour fields, pairs each latent pixel with the colour it covers, and solves for weights
    plus bias. Exact for whatever VAE the model ships, so no per-model constants.
    """
    g = torch.Generator().manual_seed(0)
    coarse = torch.rand(16, 3, 4, 4, generator=g)
    images = torch.nn.functional.interpolate(coarse, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
    # With offloading the VAE sits on the CPU and moves to `device` only while it runs, so its input goes there too.
    with torch.no_grad():
        z = vae.encode((images * 2 - 1).to(device, vae.dtype)).latent_dist.mean.float().cpu()
    shift = getattr(vae.config, "shift_factor", None) or 0.0
    z = (z - shift) * vae.config.scaling_factor  # what the transformer sees and the step callback hands us
    colours = torch.nn.functional.interpolate(images, size=z.shape[-2:], mode="area")
    x = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    y = colours.permute(0, 2, 3, 1).reshape(-1, 3)
    return torch.linalg.lstsq(x, y).solution  # (channels + 1, 3)


def write_preview(latents, projection, path):
    z = latents[0].float().permute(1, 2, 0)  # h, w, channels
    rgb = torch.cat([z, torch.ones(*z.shape[:2], 1, device=z.device)], dim=2) @ projection.to(z.device)
    pixels = (rgb.clamp(0, 1) * 255).byte().cpu().numpy()
    image = Image.fromarray(pixels)
    image.resize((image.width * 2, image.height * 2), Image.BILINEAR).save(path, quality=80)


def main():
    device, fallback = pick(torch)
    started = time.time()
    try:
        pipe = load(device)
        projection = fit_preview(pipe.vae, pipe._execution_device)
    except Exception as error:  # the core shows this instead of "the model stopped"
        kind = "oom" if isinstance(error, torch.OutOfMemoryError) else "model"
        send(event="error", kind=kind, message=f"{type(error).__name__}: {error}".split("\n")[0])
        raise
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    send(event="ready", family=FAMILY, device=device, fallback=fallback, elapsed=round(time.time() - started, 1),
         vram=round(torch.cuda.memory_allocated() / GIB, 2) if device == "cuda" else 0.0)

    requests = queue.Queue()
    cancelled = set()
    current = {"id": None}
    stop = threading.Event()

    def read():
        """Own thread, so a cancel lands while a generate is running."""
        for line in sys.stdin:
            line = line.lstrip("﻿").strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("op") == "cancel":
                cancelled.add(r["id"])
                if current["id"] == r["id"]:
                    stop.set()
            else:
                requests.put(r)
        requests.put(None)  # stdin closed: Latent is gone

    threading.Thread(target=read, daemon=True).start()

    while (r := requests.get()) is not None:
        rid = r["id"]
        if rid in cancelled:
            send(event="cancelled", id=rid)
            continue
        current["id"] = rid
        stop.clear()
        try:
            send(**generate(pipe, projection, device, r, stop))
        except Cancelled:
            send(event="cancelled", id=rid)
        except torch.OutOfMemoryError as error:
            send(event="error", id=rid, kind="oom", message=str(error).split("\n")[0])
        except Exception as error:  # report and keep the model loaded
            send(event="error", id=rid, kind="model", message=f"{type(error).__name__}: {error}")
        finally:
            current["id"] = None
            pipe.maybe_free_model_hooks()  # a run that stopped early leaves offload hooks mid-cycle
            if device == "cuda":
                torch.cuda.empty_cache()


def generate(pipe, projection, device, r, stop):
    if r.get("sources"):
        raise ValueError("this model makes new images; pick a model that can edit")
    rid = r["id"]
    options = r.get("options") or {}
    steps = int(options.get("steps", 9))
    preview = r["output"] + ".p.jpg"
    started = time.time()
    last = [started]

    def on_step(pipeline, i, t, tensors):
        if stop.is_set():
            raise Cancelled()
        write_preview(tensors["latents"], projection, preview)
        now = time.time()
        send(event="progress", id=rid, step=i + 1, steps=steps, preview=preview, sec_per_step=round(now - last[0], 2))
        last[0] = now
        return {}

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    image = pipe(
        prompt=r["prompt"],
        width=int(r["width"]),
        height=int(r["height"]),
        num_inference_steps=steps,
        guidance_scale=float(options.get("guidance", 0.0)),
        generator=torch.Generator("cpu").manual_seed(int(r["seed"])),  # CPU generator: same seed, same image, offloaded or not
        callback_on_step_end=on_step,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]

    info = PngInfo()
    info.add_text("parameters", json.dumps(r.get("recipe") or {}))
    info.add_text("Software", "Latent")
    info.add_text("ai_generated", "true")
    image.save(r["output"], pnginfo=info)
    thumb = r["output"] + ".t.jpg"
    small = image.copy()
    small.thumbnail((384, 384))
    small.save(thumb, quality=85)
    peak = round(torch.cuda.max_memory_allocated() / GIB, 2) if device == "cuda" else 0.0
    return dict(event="done", id=rid, output=r["output"], thumb=thumb, seed=int(r["seed"]),
                elapsed=round(time.time() - started, 2), peak_vram=peak)


if __name__ == "__main__":
    main()
