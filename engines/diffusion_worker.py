"""Diffusion worker for Latent. Run with diffusion-env python (-X utf8). Speaks JSON lines on stdin/stdout.

Start args: <family> <model folder> <offload: none | text | model | stream | disk> <weights: bf16 | fp8> [<base model folder>]
  base:  the folder of the model this one runs on (its text encoder and VAE), for models that need one
  none:  everything on the GPU
  text:  the text encoder stays on the CPU (prompts are encoded once and cached); the rest stays on the GPU
  model: each part moves to the GPU only while it runs (least VRAM, slowest: weights cross PCIe every frame)
  stream: the transformer and text encoder wait in RAM and visit the GPU one block at a time (a few GB of VRAM, much slower)
  disk:  the same, but they wait on disk in LATENT_OFFLOAD_DIR, so RAM only holds a block (slowest)
  fp8:   the transformer's weights are stored as FP8 (half the memory) and computed in bf16 a layer at a time
Requests, one at a time (docs/PLAN.md section 5):
  {"op": "generate", "id", "prompt", "negative"?, "sources": [paths], "width", "height", "seed",
   "options": {"steps"?}, "output", "recipe"}
  {"op": "cancel", "id"}      may arrive while a generate runs; stops it at the next step, model stays loaded
Events: ready, progress, done, cancelled, error.
"""
import json
import os
import queue
import sys
import tempfile
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
WEIGHTS = sys.argv[4] if len(sys.argv) > 4 else "bf16"
BASE = Path(sys.argv[5]) if len(sys.argv) > 5 else None
# Where "disk" streaming keeps the weights while a model is loaded; Latent clears it when the model unloads.
OFFLOAD_DIR = Path(os.environ.get("LATENT_OFFLOAD_DIR") or tempfile.gettempdir()) / "latent-offload"
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


def store(transformer):
    """FP8 storage when the variant asks for it: half the VRAM, computed in bf16 a layer at a time (diffusers)."""
    if WEIGHTS == "fp8":
        transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)


def streaming():
    """Layer streaming, for PCs too small for anything else: the big parts wait in RAM ("stream") or on disk ("disk") and
    visit the GPU one block at a time, so a few GB of VRAM is enough. Slow: every step moves the whole model."""
    return OFFLOAD in ("stream", "disk")


def stream(module, name, device):
    """Start streaming one part (diffusers' group offloading). On disk each part gets its own folder: parts that share
    one overwrite each other's weights."""
    import gc
    from diffusers.hooks import apply_group_offloading
    where = {"offload_to_disk_path": str(OFFLOAD_DIR / name)} if OFFLOAD == "disk" else {}
    # Blocks are only found as direct children of the module, and a text encoder nests its layers (a Qwen3-VL's are at
    # model.language_model.layers): left to blocks it would go to the GPU whole, 17 GB of it. Per layer works for any nesting.
    how = {"offload_type": "leaf_level"} if name == "text_encoder" else {"offload_type": "block_level", "num_blocks_per_group": 1}
    apply_group_offloading(module, onload_device=torch.device(device), offload_device=torch.device("cpu"), **how, **where)
    gc.collect()  # the weights just left RAM (to the disk) or moved into the offload cache; free what's left of the load


def pipeline(cls, folder, dtype, device, **given):
    """cls.from_pretrained. When streaming, the transformer and text encoder load one at a time and start streaming at
    once, so RAM never holds more than the largest one."""
    if not streaming():
        return cls.from_pretrained(folder, dtype=dtype, **given)
    import importlib
    index = json.loads((folder / "model_index.json").read_text())
    parts = [n for n in ("transformer", "text_encoder") if n in index and n not in given]
    pipe = cls.from_pretrained(folder, dtype=dtype, **given, **{n: None for n in parts})
    for n in parts:
        library, class_name = index[n]
        module = getattr(importlib.import_module(library), class_name).from_pretrained(folder / n, dtype=dtype)
        stream(module, n, device)
        pipe.register_modules(**{n: module})
    return pipe


def run_on_cpu(text_encoder):
    """Keep a text encoder in RAM and run it there: its inputs move to the CPU and its outputs back to the GPU, so the
    pipeline around it doesn't notice. For pipelines whose run device is read from the VAE, which stays on the GPU."""
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module
    text_encoder.to("cpu")
    add_hook_to_module(text_encoder, AlignDevicesHook(execution_device="cpu", io_same_device=True))


def decode_small(vae, latents):
    """Decode latents with the model's own VAE, on whatever device the VAE is on."""
    with torch.no_grad():
        return vae.decode(latents.to(next(vae.parameters()).device, vae.dtype)).sample


def to_image(pixels):
    """(3, h, w) in -1..1 -> PIL image."""
    return Image.fromarray(((pixels.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy())


def flux2_preview(vae, latents, width, height):
    """FLUX.2-VAE models (FLUX.2 and FLUX.3 klein) denoise packed tokens: 2 x 2 patches of VAE latents, normalised with
    the VAE's batch-norm statistics. Their latents aren't close to linear in colour (13 % error for a fitted
    projection, measured), so the preview undoes the packing and decodes 4x smaller for real: about 13 ms at 1024 px."""
    b, _, c = latents.shape
    h, w = height // 16, width // 16
    z = latents.view(b, h, w, c).permute(0, 3, 1, 2)
    bn = vae.bn
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(z)
    z = z * std + bn.running_mean.view(1, -1, 1, 1).to(z)
    z = z.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)
    return to_image(decode_small(vae, torch.nn.functional.avg_pool2d(z, 4))[0])


# ---------------------------------------------------------------- model families
# Each family loads itself for OFFLOAD, encodes prompts (cached), runs a frame calling on_step(i, latents) after
# every step, and turns a step's latents into a small preview image.

class ZImage:
    default_steps = 9

    def __init__(self, device, dtype):
        from diffusers import ZImagePipeline
        pipe = pipeline(ZImagePipeline, FOLDER, dtype, device)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_tiling()  # decoding in tiles keeps the end-of-frame peak from pushing VRAM into shared memory
        encoder, encode_on = pipe.text_encoder, device
        if device == "cpu" or OFFLOAD == "none":
            pipe.to(device)
        elif streaming():
            pipe.vae.to(device)  # the transformer and text encoder are already streaming
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
        self.projection = self.fit_projection()

    def fit_projection(self):
        """Least-squares map from latent channels to RGB, fitted on this VAE: a preview costs one matrix multiply.

        Encodes smooth random colour fields, moves them into the space the transformer works in, pairs each latent
        position with the colour it covers, and solves for weights plus bias.
        """
        g = torch.Generator().manual_seed(0)
        coarse = torch.rand(16, 3, 4, 4, generator=g)
        images = torch.nn.functional.interpolate(coarse, size=(128, 128), mode="bicubic", align_corners=False).clamp(0, 1)
        # With offloading the VAE sits on the CPU and moves to the run device only while it runs; its input goes there.
        with torch.no_grad():
            z = self.vae.encode((images * 2 - 1).to(self.device, self.vae.dtype)).latent_dist.mean.float().cpu()
        shift = getattr(self.vae.config, "shift_factor", None) or 0.0
        z = (z - shift) * self.vae.config.scaling_factor  # what the transformer sees and the step callback hands us
        colours = torch.nn.functional.interpolate(images, size=z.shape[-2:], mode="area")
        x = z.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
        x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
        y = colours.permute(0, 2, 3, 1).reshape(-1, 3)
        return torch.linalg.lstsq(x, y).solution  # (channels + 1, 3)

    def preview(self, latents, width, height):
        z = latents[0].float().permute(1, 2, 0)  # h, w, channels
        rgb = torch.cat([z, torch.ones(*z.shape[:2], 1, device=z.device)], dim=2) @ self.projection.to(z.device)
        return Image.fromarray((rgb.clamp(0, 1) * 255).byte().cpu().numpy())

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
        if streaming():
            stream(text_encoder, "text_encoder", device)
        tokenizer = Qwen3VLProcessor.from_pretrained(FOLDER / "text_encoder")
        transformer = Flux3Transformer2DModel.from_pretrained(FOLDER / "transformer", dtype=dtype)
        store(transformer)
        if streaming():
            stream(transformer, "transformer", device)
        vae = AutoencoderKLFlux2.from_pretrained(FOLDER / "vae", dtype=dtype)
        vae.enable_tiling()

        blocks = Flux3DistilledAutoBlocks()
        text_blocks = blocks.sub_blocks.pop("text_encoder")
        blocks.sub_blocks.pop("prompt_upsampler")
        self.text = text_blocks.init_pipeline()
        self.text.update_components(text_encoder=text_encoder, tokenizer=tokenizer)
        self.pipe = blocks.init_pipeline()
        self.pipe.update_components(transformer=transformer, vae=vae)

        if streaming():
            vae.to(device)  # the text encoder and transformer are already streaming
        elif device != "cpu" and OFFLOAD == "model":
            text_encoder.to(device)
            self.pipe.enable_model_cpu_offload(device=device)
        else:
            text_encoder.to("cpu" if OFFLOAD == "text" or device == "cpu" else device)
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

    def preview(self, latents, width, height):
        return flux2_preview(self.vae, latents, width, height)

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


class Flux2Klein:
    """FLUX.2 klein 4B: generation and editing, step-distilled to 4 steps at guidance 1."""

    default_steps = 4

    def __init__(self, device, dtype):
        from diffusers import Flux2KleinPipeline
        pipe = pipeline(Flux2KleinPipeline, FOLDER, dtype, device)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_tiling()
        store(pipe.transformer)
        if device == "cpu" or OFFLOAD == "none":
            pipe.to(device)
        elif streaming():
            pipe.vae.to(device)
        elif OFFLOAD == "text":
            pipe.transformer.to(device)
            pipe.vae.to(device)
            run_on_cpu(pipe.text_encoder)
        else:
            pipe.enable_model_cpu_offload(device=device)
        self.pipe, self.vae, self.device = pipe, pipe.vae, pipe._execution_device
        self.prompts = PromptCache(lambda prompt: pipe.encode_prompt(prompt, device=self.device)[0])

    def preview(self, latents, width, height):
        return flux2_preview(self.vae, latents, width, height)

    def run(self, r, sources, steps, generator, on_step):
        def callback(pipeline, i, t, tensors):
            on_step(i, tensors["latents"])
            return {}

        return self.pipe(
            prompt_embeds=self.prompts(r["prompt"]), image=[Image.open(p).convert("RGB") for p in sources] or None,
            width=int(r["width"]), height=int(r["height"]), num_inference_steps=steps, guidance_scale=1.0,
            generator=generator, callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

    def release(self):
        self.pipe.maybe_free_model_hooks()


class QwenImage21:
    """Qwen-Image 2.1 Turbo: generation and editing on its saved 8-step schedule. The 7B transformer is stored as FP8
    on the GPU; the 17.5 GB text encoder runs on the CPU with the prompt and any reference image (an edit's
    embeddings depend on the image, so they aren't cached)."""

    default_steps = 8
    # Its latents need a 2x (not 4x) shrink to decode into something recognisable, which costs ~150 ms (about 10 % of
    # a step, measured), so it previews every other step to stay near 5 %.
    preview_every = 2

    def load_transformer(self, dtype):
        from diffusers import QwenImage21Transformer2DModel
        return QwenImage21Transformer2DModel.from_pretrained(FOLDER / "transformer", dtype=dtype)

    def load_pipeline(self, transformer, dtype):
        from diffusers import QwenImage21Pipeline
        return pipeline(QwenImage21Pipeline, FOLDER, dtype, self.run_device, transformer=transformer)

    def __init__(self, device, dtype):
        import gc
        self.run_device = device
        # The transformer loads and shrinks to FP8 before the text encoder arrives, so RAM never holds both at bf16.
        transformer = self.load_transformer(dtype)
        store(transformer)
        if streaming():
            stream(transformer, "transformer", device)
        elif device != "cpu":
            transformer.to(device)
        gc.collect()
        pipe = self.load_pipeline(transformer, dtype)
        pipe.set_progress_bar_config(disable=True)
        pipe.vae.enable_tiling()  # an edit encodes the reference and decodes the result; whole, that peaked at 15.9 GB
        pipe.vae.to(device)
        if device == "cpu" or OFFLOAD == "none":
            pipe.text_encoder.to(device)
        elif not streaming():
            run_on_cpu(pipe.text_encoder)
        self.pipe, self.vae, self.device = pipe, pipe.vae, pipe._execution_device

    def preview(self, latents, width, height):
        pipe, vae = self.pipe, self.vae
        z = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)  # (batch, channels, 1, h, w)
        shape = (1, vae.config.z_dim, 1, 1, 1)
        z = z * torch.tensor(vae.config.latents_std).view(shape).to(z) + torch.tensor(vae.config.latents_mean).view(shape).to(z)
        z = torch.nn.functional.avg_pool3d(z, (1, 2, 2))
        with torch.no_grad():
            pixels = vae.decode(z.to(next(vae.parameters()).device, vae.dtype), return_dict=False)[0][:, :, 0]
        return to_image(pixels[0])

    def run(self, r, sources, steps, generator, on_step):
        def callback(pipeline, i, t, tensors):
            on_step(i, tensors["latents"])
            return {}

        return self.pipe(
            prompt=r["prompt"], image=[Image.open(p).convert("RGB") for p in sources] or None,
            width=int(r["width"]), height=int(r["height"]), generator=generator, num_inference_steps=steps,
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

    def release(self):
        pass


class QwenImage21Uncensored(QwenImage21):
    """Qwen-Image 2.1 Uncensored: the base (not Turbo) weights, quantized to FP8 by their maker, run on Qwen-Image 2.1
    Turbo's text encoder, VAE and processor (the same files as the base model's, which is why that model is required).
    40 steps without guidance, as the base model's card runs it."""

    default_steps = 40
    # Qwen/Qwen-Image-2.1's scheduler_config.json: it shifts its schedule with the image size, and has no saved sigmas.
    SCHEDULER = dict(
        base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=8192, max_shift=0.9,
        num_train_timesteps=1000, shift=1.0, shift_terminal=0.02, stochastic_sampling=False, time_shift_type="exponential",
        use_beta_sigmas=False, use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)

    def load_transformer(self, dtype):
        """The file is in ComfyUI's scaled-FP8 format: each quantized weight has a per-tensor weight_scale. Dequantize
        to bf16 here; store() shrinks the weights to FP8 again the way every other model's are."""
        from accelerate import init_empty_weights
        from diffusers import QwenImage21Transformer2DModel
        from safetensors import safe_open
        with init_empty_weights():  # the weights come from the file; buffers (rotary tables) are still built for real
            transformer = QwenImage21Transformer2DModel.from_config(QwenImage21Transformer2DModel.load_config(BASE / "transformer"))
        state = {}
        with safe_open(FOLDER / "qwen-image-2.1-UC-fp8.safetensors", "pt") as f:
            for key in f.keys():
                if key.endswith((".weight_scale", ".comfy_quant")) or key == "scaled_fp8":
                    continue
                weight = f.get_tensor(key)
                if weight.dtype == torch.float8_e4m3fn:
                    weight = weight.float() * f.get_tensor(key[:-len("weight")] + "weight_scale")
                state[key] = weight.to(dtype)
        transformer.load_state_dict(state, assign=True)
        return transformer

    def load_pipeline(self, transformer, dtype):
        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
        turbo = pipeline(QwenImage21Pipeline, BASE, dtype, self.run_device, transformer=transformer)
        # A new pipeline around the same parts: Turbo's own would run its saved 8-step sigmas.
        return QwenImage21Pipeline(**{**turbo.components, "scheduler": FlowMatchEulerDiscreteScheduler(**self.SCHEDULER)})


class SanaSprint:
    """SANA-Sprint 1.6B: a small, fast generator (1 to 4 steps). Its 32x autoencoder makes the latents tiny, so the
    preview decodes them whole."""

    default_steps = 2

    def __init__(self, device, dtype):
        import inspect
        from diffusers import SanaSprintPipeline
        pipe = pipeline(SanaSprintPipeline, FOLDER, dtype, device)
        pipe.set_progress_bar_config(disable=True)
        encoder, encode_on = pipe.text_encoder, device
        if device == "cpu" or OFFLOAD == "none":
            pipe.to(device)
        elif streaming():
            pipe.vae.to(device)
        else:
            encode_on = "cpu"
            pipe.text_encoder = None  # out of the pipeline, so it runs on the transformer's device
            pipe.transformer.to(device)
            pipe.vae.to(device)
        self.pipe, self.vae, self.device = pipe, pipe.vae, pipe._execution_device
        # The pipeline expands prompts with these instructions by default; embeddings made here must include them.
        instruction = inspect.signature(SanaSprintPipeline.__call__).parameters["complex_human_instruction"].default

        def encode(prompt):
            pipe.text_encoder = encoder
            try:
                embeds, mask = pipe.encode_prompt(prompt, device=encode_on, complex_human_instruction=instruction)
            finally:
                if encode_on == "cpu" and device != "cpu":
                    pipe.text_encoder = None
            return embeds.to(self.device), mask.to(self.device)

        self.prompts = PromptCache(encode)

    def preview(self, latents, width, height):
        return to_image(decode_small(self.vae, latents / self.vae.config.scaling_factor)[0])

    def run(self, r, sources, steps, generator, on_step):
        if sources:
            raise ValueError("SANA-Sprint makes new images; pick a model that can edit")

        def callback(pipeline, i, t, tensors):
            on_step(i, tensors["latents"])
            return {}

        embeds, mask = self.prompts(r["prompt"])
        return self.pipe(
            prompt_embeds=embeds, prompt_attention_mask=mask, width=int(r["width"]), height=int(r["height"]),
            num_inference_steps=steps, generator=generator,
            callback_on_step_end=callback, callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

    def release(self):
        pass


class Kromoss:
    """Kromoss, a 5B MOSS-Image fine-tune, run the way its card recommends for speed: the DMD2 turbo LoRA at 8 steps and
    guidance 1, so there is no negative branch. The pipeline is the vendored pipeline_moss_image.py, not repo code.
    The 9B text encoder runs on the CPU."""

    default_steps = 8

    def __init__(self, device, dtype):
        import gc
        from diffusers import AutoencoderKLFlux2
        from pipeline_moss_image import MOSSImagePipeline, MOSSImageScheduler, MOSSImageTransformer
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer
        # The transformer goes to the GPU (as FP8 when asked) before the text encoder arrives, so RAM never holds both.
        transformer = MOSSImageTransformer.from_pretrained(FOLDER / "transformer", torch_dtype=dtype).eval()
        store(transformer)
        if streaming():
            stream(transformer, "transformer", device)
        elif device != "cpu":
            transformer.to(device)
        gc.collect()
        # The LoRA is applied unmerged, as its card says: its delta is close to bf16's rounding step.
        lora = load_file(FOLDER / "8-steps-dmd2-lora.safetensors", device=device if device != "mps" else "cpu")
        modules = dict(transformer.named_modules())
        for key in [k for k in lora if k.endswith(".lora_down.weight")]:
            name = key[len("diffusion_model."):-len(".lora_down.weight")]
            down, up = lora[key].float(), lora[key.replace("lora_down", "lora_up")].float()
            scale = lora[f"diffusion_model.{name}.alpha"].item() / down.shape[0]
            modules[name].register_forward_hook(
                lambda module, args, out, down=down, up=up, scale=scale: out + (scale * (args[0].float() @ down.T) @ up.T).to(out.dtype))
        text_encoder = AutoModel.from_pretrained(FOLDER / "text_encoder", dtype=dtype).eval()
        if streaming():
            stream(text_encoder, "text_encoder", device)
            # Its linear-attention layers read their conv's weight directly, which no per-layer hook sees: keep those (tiny) here.
            for conv in (m for m in text_encoder.modules() if isinstance(m, torch.nn.Conv1d)):
                conv.to(device)
        vae = AutoencoderKLFlux2.from_pretrained(FOLDER, subfolder="vae", torch_dtype=torch.float32).eval()
        pipe = MOSSImagePipeline(transformer, MOSSImageScheduler(), text_encoder, AutoTokenizer.from_pretrained(FOLDER / "tokenizer"), vae)
        pipe.set_progress_bar_config(disable=True)
        vae.to(device)
        if device == "cpu" or OFFLOAD == "none":
            text_encoder.to(device)
        elif not streaming():
            run_on_cpu(text_encoder)
        self.pipe, self.vae, self.device = pipe, vae, pipe._execution_device
        self.prompts = PromptCache(lambda prompt: pipe.encode_prompt(prompt))

    def preview(self, latents, width, height):
        # Unpacked latents (batch, channels, h, w), where the FLUX.2 VAE models above have packed tokens.
        b, c, h, w = latents.shape
        return flux2_preview(self.vae, latents.permute(0, 2, 3, 1).reshape(b, h * w, c), width, height)

    def run(self, r, sources, steps, generator, on_step):
        if sources:
            raise ValueError("Kromoss makes new images; pick a model that can edit")
        embeds, mask = self.prompts(r["prompt"])
        return self.pipe(
            prompt_embeds=embeds, prompt_attention_mask=mask, width=int(r["width"]), height=int(r["height"]),
            num_inference_steps=steps, guidance_scale=1.0, generator=generator, callback_on_step_end=on_step,
        ).images[0]

    def release(self):
        pass


FAMILIES = {"zimage": ZImage, "flux3klein": Flux3Klein, "flux2klein": Flux2Klein, "qwenimage21": QwenImage21,
            "sanasprint": SanaSprint, "kromoss": Kromoss, "qwenimage21uc": QwenImage21Uncensored}


# ---------------------------------------------------------------- previews

def write_preview(image, path):
    scale = 256 / max(image.size)  # previews are 256 px on the long side, whatever the model makes them at
    image = image.convert("RGB")  # JPEG has no alpha; Qwen-Image 2.1 decodes RGBA (it can make transparent images)
    image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR).save(path, quality=80)


# ---------------------------------------------------------------- protocol

def main():
    global OFFLOAD
    device, fallback = pick(torch)
    if device == "cpu" and streaming():
        OFFLOAD = "none"  # nothing to stream to: the model just runs from RAM
    started = time.time()
    try:
        dtype = torch.bfloat16
        model = FAMILIES[FAMILY](device, dtype)
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
        # Latent is gone (its end of stdin closed): exit now, even mid-frame, so no worker keeps the GPU.
        os._exit(0)

    threading.Thread(target=read, daemon=True).start()

    while (r := requests.get()) is not None:
        rid = r["id"]
        if rid in cancelled:
            send(event="cancelled", id=rid)
            continue
        current["id"] = rid
        stop.clear()
        try:
            send(**generate(model, device, r, stop))
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


def generate(model, device, r, stop):
    rid = r["id"]
    steps = int((r.get("options") or {}).get("steps", model.default_steps))
    width, height = int(r["width"]), int(r["height"])
    preview = r["output"] + ".p.jpg"
    started = time.time()
    last = [started]
    shown = [None]  # the preview path once a preview exists; until then the UI keeps its placeholder

    def on_step(i, latents):
        if stop.is_set():
            raise Cancelled()
        try:
            if (i + 1) % getattr(model, "preview_every", 1) == 0 or i + 1 == steps:
                write_preview(model.preview(latents, width, height), preview)
                shown[0] = preview
        except Exception as error:  # a preview is cosmetic: log it, never lose the frame over it
            print(f"preview failed at step {i + 1}: {type(error).__name__}: {error}", file=sys.stderr)
        now = time.time()
        send(event="progress", id=rid, step=i + 1, steps=steps, preview=shown[0], sec_per_step=round(now - last[0], 2))
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
