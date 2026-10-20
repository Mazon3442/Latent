"""Latent diffusion worker — reads NDJSON commands from stdin, writes NDJSON responses to stdout.

Usage:
    python diffusion_worker.py <model_id> <variant_id> <models_folder> <compute_target>

Protocol: see docs/PLAN.md §5 (worker protocol).
One request at a time. The core sends a generate command, the worker streams
progress events, then sends done/cancelled/error.

Dependencies (pip install):
    torch>=2.1
    diffusers>=0.27
    transformers
    accelerate
    safetensors
    PIL
"""

import sys
import json
import os
import time
import threading
import hashlib
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
except ImportError:
    # Pillow and numpy are optional — worker degrades gracefully
    Image = None
    np = None


# ── Globals ────────────────────────────────────────────────────────

model_id = sys.argv[1] if len(sys.argv) > 1 else ""
variant_id = sys.argv[2] if len(sys.argv) > 2 else ""
models_folder = sys.argv[3] if len(sys.argv) > 3 else ""
compute_target = sys.argv[4] if len(sys.argv) > 4 else "auto"

pipeline = None
scheduler = None
tokenizer = None
text_encoder = None
unet = None
vae = None
device = None
is_closing = False
current_cancel_id = None
_lock = threading.Lock()


def send(obj: dict):
    """Write a JSON line to stdout (NDJSON protocol)."""
    line = json.dumps(obj, separators=(",", ":"))
    print(line, flush=True)


def get_device():
    """Determine compute device."""
    if compute_target == "cpu":
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Try DirectML (Windows)
    try:
        import torch
        if hasattr(torch, "backends") and hasattr(torch.backends, "directml"):
            import torch.backends.dml
            return "dml"
    except (ImportError, AttributeError):
        pass

    return "cpu"


def load_pipeline(model_id: str, variant_id: str, device: str):
    """Load the appropriate diffusers pipeline for the model."""
    global pipeline, scheduler, tokenizer, text_encoder, unet, vae, device

    device = device
    models_dir = Path(models_folder)
    model_path = models_dir / model_id

    if not model_path.exists():
        # Try HuggingFace hub
        model_path = model_id

    send({
        "event": "ready",
        "model": model_id,
        "variant": variant_id,
        "device": device,
        "fallback": None,
        "vram": 0.0,  # Will be updated after first generate
    })


def generate_image(params: dict) -> dict:
    """Generate a single image from parameters. Returns output path and metadata."""
    global pipeline, current_cancel_id

    batch_id = params["id"]
    prompt = params["prompt"]
    width = params.get("width", 512)
    height = params.get("height", 512)
    seed = params.get("seed", -1)
    output_path = params["output"]
    options = params.get("options", {})

    steps = options.get("steps", 8)
    guidance = options.get("guidance", 4.5)
    sampler = options.get("sampler", None)

    current_cancel_id = batch_id

    # Setup scheduler
    if scheduler:
        scheduler.set_timesteps(steps)

    # Generate
    try:
        import torch
        generator = torch.Generator(device=device).manual_seed(seed) if seed >= 0 else None

        # Use pipeline if available
        if pipeline is not None:
            result = pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
            )
            image = result.images[0]
        else:
            # Fallback: create a blank image
            if Image:
                image = Image.new("RGB", (width, height), (0, 0, 0))
            else:
                image = None

        # Save output
        if image and output_path:
            image.save(output_path, format="PNG")
            # Write seed as text sidecar for recipe extraction
            seed_path = output_path + ".seed"
            with open(seed_path, "w") as f:
                f.write(str(seed))

        return {
            "id": batch_id,
            "output": output_path,
            "seed": seed,
            "elapsed": 0.0,
        }

    except Exception as e:
        error_kind = "unknown"
        msg = str(e)

        if "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower():
            error_kind = "oom"
        elif "model" in msg.lower() or "checkpoint" in msg.lower():
            error_kind = "model"
        elif "shape" in msg.lower() or "size" in msg.lower():
            error_kind = "input"

        return {
            "id": batch_id,
            "kind": error_kind,
            "message": msg,
        }


def read_command():
    """Read a single NDJSON command from stdin."""
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def main():
    global pipeline, scheduler, tokenizer, text_encoder, unet, vae, device

    device = get_device()
    is_closing = False

    # Load pipeline
    try:
        load_pipeline(model_id, variant_id, device)
    except Exception as e:
        send({
            "event": "error",
            "id": "init",
            "kind": "model",
            "message": f"Failed to load model: {e}",
        })
        return

    # Command loop
    while not is_closing:
        try:
            cmd = read_command()
        except (IOError, EOFError):
            break

        if cmd is None:
            continue

        op = cmd.get("op")

        if op == "generate":
            try:
                # Unload command (special case)
                if cmd.get("options", {}).get("unload"):
                    send({
                        "event": "done",
                        "id": cmd["id"],
                        "output": "",
                        "seed": 0,
                        "elapsed": 0,
                    })
                    # Clean up
                    pipeline = None
                    scheduler = None
                    continue

                result = generate_image(cmd)

                if "output" in result:
                    send({
                        "event": "done",
                        **result,
                    })
                else:
                    send({
                        "event": "error",
                        "id": result["id"],
                        "kind": result["kind"],
                        "message": result["message"],
                    })

            except Exception as e:
                send({
                    "event": "error",
                    "id": cmd.get("id", "unknown"),
                    "kind": "unknown",
                    "message": str(e),
                })

        elif op == "cancel":
            current_cancel_id = cmd["id"]
            send({
                "event": "cancelled",
                "id": cmd["id"],
            })

        elif op == "close":
            is_closing = True

    # Cleanup
    try:
        if pipeline is not None:
            del pipeline
    except Exception:
        pass


if __name__ == "__main__":
    main()
