"""Diffusion worker for Latent. Run with diffusion-env python (-X utf8). Speaks JSON lines on stdin/stdout.

Start args: <family> <model folder> <offload: none | text | model>
  none:  everything on the GPU
  text:  the text encoder stays on the CPU (prompts are encoded once and cached); the rest stays on the GPU
  model: each part moves to the GPU only while it runs (least VRAM, slowest: weights cross PCIe every frame)
Requests, one at a time (docs/PLAN.md section 5):
  {"op": "generate", "id", "prompt", "negative"?, "sources": [paths], "width", "height", "seed",
   "options": {"steps"?}, "output", "recipe"}
  {"op": "cancel", "id"}      may arrive while a generate runs; stops it at the next step, model stays loaded
Events: ready, progress, done, cancelled, error.
"""
import json
import queue
import sys
import threading
import time
from pathlib import Path

real_out = sys.stdout
sys.stdout = sys.stderr  # library prints must not corrupt the protocol stream
sys.stdin.reconfigure(encoding="utf-8")

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from device import pick

FAMILY = sys.argv[1]
FOLDER = Path(sys.argv[2])
OFFLOAD = sys.argv[3] if len(sys.argv) > 3 else "none"
GIB = 1024 ** 3

_out = threading.Lock()


def send(**message):
    with _out:
        real_out.write(json.dumps(message) + "\n")
        real_out.flush()


class Cancelled(Exception):
    pass


class PromptCache:
    """Prompt -> embeddings on the run device. Encoding is the slow part on the CPU, and batches reuse one prompt."""

    def __init__(self, encode):
        self.encode, self.items = encode, {}

    def __call__(self, prompt):
        if prompt not in self.items:
            if len(self.items) >= 16:
                self.items.pop(next(iter(self.items)))
            with torch.no_grad():
                self.items[prompt] = self.encode(prompt)
        return self.items[prompt]


# ---------------------------------------------------------------- model families
# Each family loads itself for OFFLOAD, and knows how its latents relate to the VAE, so previews work the same way.

class ZImage:
    default_steps = 9

    def __init__(self, device, dtype):
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(FOLDER, dtype=dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_tiling()  # decoding in tiles keeps the end-of-frame peak from pushing VRAM into shared memory
        encoder, encode_on = pipe.text_encoder, device
        if device == "cpu" or OFFLOAD == "none":
            pipe.to(device)
        elif OFFLOAD == "text":
            encode_on = "cpu"
            pipe.text_encoder = None  # out of the pipeline, so it runs on the transformer's device
            pipe.transformer.to(device)
            pipe.vae.to(device)
        else:
            pipe.enable_model_cpu_offload(device=device)
        self.pipe, self.vae, self.device = pipe, pipe.vae, pipe._execution_device

        def encode(prompt):
            pipe.text_encoder = encoder
            try:
                embeds, _ = pipe.encode_prompt(prompt, device=encode_on, do_classifier_free_guidance=False)
            finally:
                if OFFLOAD == "text" and device != "cpu":
                    pipe.text_encoder = None
            return [e.to(self.device) for e in embeds]

        self.prompts = PromptCache(encode)

    def model_space(self, z):
        shift = getattr(self.vae.config, "shift_factor", None) or 0.0
        return (z - shift) * self.vae.config.scaling_factor

    def grid(self, latents, width, height):
        return latents  # already (batch, channels, height / 8, width / 8)

    def run(self, r, sources, steps, generator, on_step):
        if sources:
            raise ValueError("Z-Image Turbo makes new images; pick a model that can edit")

        def callback(pipeline, i, t, tensors):
            on_step(i, tensors["latents"])
            return {}

        return self.pipe(
            prompt_embeds=self.prompts(r["prompt"]), width=int(r["width"]), height=int(r["height"]),
            num_inference_steps=steps, guidance_scale=0.0, generator=generator,
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

    def release(self):
        self.pipe.maybe_free_model_hooks()  # a run that stopped early leaves offload hooks mid-cycle


class Flux3Klein:
    """FLUX.3 klein through diffusers' modular blocks: prompt upsampler (skipped) -> text encoder -> VAE encoder
    (reference images) -> denoise -> decode. The text encoder runs as its own pipeline, so it can stay on the CPU."""

    default_steps = 16

    def __init__(self, device, dtype):
        from diffusers import AutoencoderKLFlux2, Flux3DistilledAutoBlocks, Flux3Transformer2DModel
        from diffusers.modular_pipelines.flux3.denoise import Flux3LoopAfterDenoiser
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        # Components come from the downloaded folders; the repo's index points at the Hub, which loading never touches.
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(FOLDER / "text_encoder", dtype=dtype)
        tokenizer = Qwen3VLProcessor.from_pretrained(FOLDER / "text_encoder")
        transformer = Flux3Transformer2DModel.from_pretrained(FOLDER / "transformer", dtype=dtype)
        vae = AutoencoderKLFlux2.from_pretrained(FOLDER / "vae", dtype=dtype)
        vae.enable_tiling()

        blocks = Flux3DistilledAutoBlocks()
        text_blocks = blocks.sub_blocks.pop("text_encoder")
        blocks.sub_blocks.pop("prompt_upsampler")
        self.text = text_blocks.init_pipeline()
        self.text.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
        self.pipe = blocks.init_pipeline()
        self.pipe.update_components(transformer=transformer, vae=vae)

        text_encoder.to("cpu" if OFFLOAD == "text" or device == "cpu" else device)
        if device != "cpu" and OFFLOAD == "model":
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            transformer.to(device)
            vae.to(device)
        self.vae, self.device = vae, torch.device(device)
        self.prompts = PromptCache(lambda prompt: self.text(prompt=prompt, output="prompt_embeds").to(self.device))

        # Modular pipelines have no step callback; every step ends in this block, so it carries the hook.
        self.on_step = None
        after = Flux3LoopAfterDenoiser.__call__
        worker = self

        def after_step(block, components, block_state, i, t):
            components, block_state = after(block, components, block_state, i, t)
            if worker.on_step:
                worker.on_step(i, block_state.latents)
            return components, block_state

        Flux3LoopAfterDenoiser.__call__ = after_step

    def model_space(self, z):
        # What the denoiser sees: 2 x 2 patches of the VAE latents, normalised with the VAE's batch-norm statistics.
        b, c, h, w = z.shape
        z = z.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)
        mean = self.vae.bn.running_mean.view(1, -1, 1, 1).float().cpu()
        std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1).float().cpu() + self.vae.config.batch_norm_eps)
        return (z - mean) / std

    def grid(self, latents, width, height):
        b, _, c = latents.shape  # packed tokens, row by row
        return latents.view(b, height // 16, width // 16, c).permute(0, 3, 1, 2)

    def run(self, r, sources, steps, generator, on_step):
        refs = [Image.open(p).convert("RGB") for p in sources]
        self.on_step = on_step
        try:
            return self.pipe(
                prompt_embeds=self.prompts(r["prompt"]), image=refs or None, height=int(r["height"]),
                width=int(r["width"]), num_inference_steps=steps, generator=generator, output="images",
            )[0]
        finally:
            self.on_step = None

    def release(self):
        pass


FAMILIES = {"zimage": ZImage, "flux3klein": Flux3Klein}


# ---------------------------------------------------------------- previews

def fit_preview(model, device):
    """Least-squares map from the denoiser's latent channels to RGB, fitted on this model's own VAE.

    Encodes smooth random colour fields, moves them into the space the denoiser works in, pairs each latent position
    with the colour it covers, and solves for weights plus bias. Exact for whatever VAE a model ships, so no
    per-model constants.
    """
    vae = model.vae
    g = torch.Generator().manual_seed(0)
    coarse = torch.rand(16, 3, 4, 4, generator=g)
    images = torch.nn.functional.interpolate(coarse, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
    # With offloading the VAE sits on the CPU and moves to `device` only while it runs, so its input goes there too.
    with torch.no_grad():
        z = vae.encode((images * 2 - 1).to(device, vae.dtype)).latent_dist.mean.float().cpu()
    z = model.model_space(z)
    colours = torch.nn.functional.interpolate(images, size=z.shape[-2:], mode="area")
    x = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    y = colours.permute(0, 2, 3, 1).reshape(-1, 3)
    return torch.linalg.lstsq(x, y).solution  # (channels + 1, 3)


def write_preview(grid, projection, path):
    z = grid[0].float().permute(1, 2, 0)  # h, w, channels
    rgb = torch.cat([z, torch.ones(*z.shape[:2], 1, device=z.device)], dim=2) @ projection.to(z.device)
    image = Image.fromarray((rgb.clamp(0, 1) * 255).byte().cpu().numpy())
    scale = 256 / max(image.size)  # previews are 256 px on the long side, whatever the latent grid
    image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR).save(path, quality=80)


# ---------------------------------------------------------------- protocol

def main():
    device, fallback = pick(torch)
    started = time.time()
    try:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32  # bfloat16 is slow or unsupported on most CPUs
        model = FAMILIES[FAMILY](device, dtype)
        projection = fit_preview(model, model.device)
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
            send(**generate(model, projection, device, r, stop))
        except Cancelled:
            send(event="cancelled", id=rid)
        except torch.OutOfMemoryError as error:
            send(event="error", id=rid, kind="oom", message=str(error).split("\n")[0])
        except Exception as error:  # report and keep the model loaded
            send(event="error", id=rid, kind="model", message=f"{type(error).__name__}: {error}")
        finally:
            current["id"] = None
            model.release()
            if device == "cuda":
                torch.cuda.empty_cache()


def generate(model, projection, device, r, stop):
    rid = r["id"]
    steps = int((r.get("options") or {}).get("steps", model.default_steps))
    width, height = int(r["width"]), int(r["height"])
    preview = r["output"] + ".p.jpg"
    started = time.time()
    last = [started]

    def on_step(i, latents):
        if stop.is_set():
            raise Cancelled()
        write_preview(model.grid(latents, width, height), projection, preview)
        now = time.time()
        send(event="progress", id=rid, step=i + 1, steps=steps, preview=preview, sec_per_step=round(now - last[0], 2))
        last[0] = now

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # CPU generator: same seed, same image, offloaded or not.
    image = model.run(r, r.get("sources") or [], steps, torch.Generator("cpu").manual_seed(int(r["seed"])), on_step)

    info = PngInfo()
    info.add_text("parameters", json.dumps(r.get("recipe") or {}))
    info.add_text("Software", "Latent")
    info.add_text("ai_generated", "true")
    image.save(r["output"], pnginfo=info)
    thumb = r["output"] + ".t.jpg"
    small = image.copy()
    small.thumbnail((384, 384))
    small.convert("RGB").save(thumb, quality=85)
    peak = round(torch.cuda.max_memory_allocated() / GIB, 2) if device == "cuda" else 0.0
    return dict(event="done", id=rid, output=r["output"], thumb=thumb, seed=int(r["seed"]),
                elapsed=round(time.time() - started, 2), peak_vram=peak)


if __name__ == "__main__":
    main()
