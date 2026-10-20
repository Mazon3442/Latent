/**
 * Latent dev mock backend — implements the Tauri command/event contract
 * from docs/PLAN.md §5 using canned data and timers.
 *
 * Usage: place in app/ui/dev/mock.js, then in index.html replace the
 * real script with <script src="dev/mock.js"></script>.
 *
 * This lets the two sides (UI + backend) develop independently.
 */
(() => {
  "use strict";

  // ── Canned model catalog ───────────────────────────────────────
  const MOCK_MODELS = [
    {
      id: "z-image-turbo",
      name: "Z-Image Turbo",
      family: "z-image",
      license: "Apache 2.0",
      licenseUrl: "",
      commercial: true,
      can: { generate: true, edit: false, inpaint: false, multiRef: false, negative: false },
      sizes: [[512, 512], [1024, 576], [1152, 640], [832, 1216]],
      grid: 16,
      options: {
        steps: { min: 4, max: 8, default: 8 },
        guidance: { min: 1.0, max: 10.0, default: 4.5 },
        sampler: ["DPM++ 2M", "DPM++ SDE", "Euler a"],
      },
      variants: [
        { id: "bf16", label: "BF16", vram_mb: 13312, ram_mb: 2048, disk_mb: 12288, quantized: false, offloaded: false, fits: "yes" },
        { id: "fp8", label: "FP8 (quantized)", vram_mb: 8192, ram_mb: 3072, disk_mb: 6144, quantized: true, offloaded: false, fits: "yes" },
      ],
      defaultVariant: "bf16",
      installed: false,
    },
    {
      id: "flux2-klein-4b",
      name: "FLUX.2 [klein] 4B",
      family: "flux2",
      license: "Apache 2.0",
      licenseUrl: "",
      commercial: true,
      can: { generate: true, edit: true, inpaint: false, multiRef: true, negative: false },
      sizes: [[1024, 1024], [1152, 896], [1216, 832], [1536, 640]],
      grid: 16,
      options: {
        steps: { min: 8, max: 50, default: 28 },
        guidance: { min: 1.0, max: 15.0, default: 6.0 },
        sampler: ["Euler", "Euler a", "DPM++ 2M", "DPM++ SDE"],
        strength: { min: 0.0, max: 1.0, default: 0.55 },
      },
      variants: [
        { id: "bf16", label: "BF16", vram_mb: 10240, ram_mb: 2048, disk_mb: 8192, quantized: false, offloaded: false, fits: "yes" },
        { id: "nf4", label: "NF4 (quantized)", vram_mb: 6144, ram_mb: 4096, disk_mb: 3072, quantized: true, offloaded: false, fits: "yes" },
      ],
      defaultVariant: "bf16",
      installed: true,
    },
  ];

  // ── State ──────────────────────────────────────────────────────
  const state = {
    settings: {
      theme: "dark",
      save_folder: "C:\\Users\\mason\\Pictures\\Latent",
      save_format: "png",
      filename_pattern: "latent_{date}_{time}_{model}_{seed}",
      warn_close: true,
      models_folder: "C:\\Users\\mason\\.local\\latent\\models",
      idle_unload_minutes: 10,
      warm_start: false,
      compute_target: "auto",
      hf_token: "",
      batch_size: 1,
      default_aspect: "1:1",
    },
    hw: {
      devices: [{
        vendor: "Nvidia",
        name: "NVIDIA GeForce RTX 5070 Ti",
        vram_total_mb: 16384,
        vram_used_mb: 0,
        driver: "560.94",
        cuda_version: "12.8",
      }],
      ram_total: 32 * 1024,
      ram_used: 8192,
      disk_free: 100 * 1024 * 1024 * 1024,
      os: "windows",
      cpu: "Intel(R) Core(TM) i9-14900K",
    },
    frames: [],
    sources: [],
    resident: null,
    nextFrameId: 1,
    busy: false,
  };

  // ── Mock Tauri bridge ──────────────────────────────────────────
  const listeners = {};
  const invokeQueue = new Map();

  window.TauriMock = true;

  window.invoke = async function (cmd, args = {}) {
    switch (cmd) {
      case "bootstrap":
        return {
          version: "0.1.0",
          settings: state.settings,
          hw: state.hw,
          models: MOCK_MODELS,
          sources: state.sources,
          frames: state.frames,
          batches: [],
          unsaved: state.frames.filter((f) => !f.saved).length,
          resident: state.resident || {},
        };

      case "save_settings": {
        const s = args.settings;
        Object.assign(state.settings, s);
        return state.settings;
      }

      case "models":
        return MOCK_MODELS;

      case "install_model": {
        const modelId = args.model_id;
        // Stream install events
        const phases = ["env", "torch", "weights", "verify", "done"];
        for (const phase of phases) {
          window._emit("install", {
            id: modelId,
            phase,
            total: phase === "weights" ? 12288 : null,
            done: phase === "done" ? 100 : null,
            message: phase === "done" ? "Complete" : undefined,
          });
          await sleep(300);
        }
        return MOCK_MODELS.find((m) => m.id === modelId);
      }

      case "remove_model": {
        const m = MOCK_MODELS.find((mod) => mod.id === args.model_id);
        if (m) m.installed = false;
        return m;
      }

      case "preload": {
        const model = args.model?.id || "flux2-klein-4b";
        const variant = args.variant || "bf16";
        window._emit("engine", { model, variant, state: "loading", elapsed: 0 });
        await sleep(2000);
        state.resident = { model, variant, state: "warm" };
        window._emit("engine", { model, variant, state: "warm", elapsed: 2.0 });
        return {};
      }

      case "unload": {
        const model = state.resident?.model;
        const variant = state.resident?.variant;
        state.resident = null;
        window._emit("engine", { state: "unloading" });
        await sleep(500);
        window._emit("engine", { state: "cold" });
        return {};
      }

      case "generate": {
        const req = args.request;
        const batchId = crypto.randomUUID().slice(0, 8);
        const count = req.count || 1;
        const steps = req.options?.steps || 8;
        const width = req.width || 512;
        const height = req.height || 512;

        // Generate mock frames
        for (let i = 0; i < count; i++) {
          const seed = req.seed ? req.seed + i : Math.floor(Math.random() * 2147483647);
          const frameId = `${batchId}-${i}`;

          // Emit progress events
          for (let step = 1; step <= steps; step++) {
            await sleep(200 + Math.random() * 300); // 200-500ms per step
            const preview = generateMockImage(width, height, seed, step, steps);
            window._emit("progress", {
              batch: batchId,
              index: i,
              step,
              steps,
              preview,
              secPerStep: (0.2 + Math.random() * 0.3).toFixed(1),
            });
          }

          // Emit done
          const outputPath = `mock://frame/${frameId}.png`;
          const finalImage = generateMockImage(width, height, seed, steps, steps);
          const frame = {
            id: frameId,
            batch: batchId,
            index: i,
            path: outputPath,
            thumb: finalImage,
            width,
            height,
            seed,
            recipe: req,
            model: req.model,
            variant: req.variant || "bf16",
            elapsed: steps * 0.35,
            saved: null,
            created_at: new Date().toISOString(),
          };
          state.frames.unshift(frame);
          window._emit("frame", frame);
        }

        // Emit batch done
        await sleep(100);
        window._emit("batch", { batch: batchId, state: "done" });

        return { batch: batchId };
      }

      case "cancel": {
        const batchId = args.batch;
        await sleep(200);
        window._emit("batch", { batch: batchId, state: "cancelled" });
        return {};
      }

      case "import_sources": {
        const paths = args.paths || [];
        const sources = paths.map((path, i) => ({
          id: `src-${state.nextFrameId++}`,
          path,
          thumb: generateMockImage(200, 200, i * 1000, 1, 1),
          width: 1920,
          height: 1080,
          name: path.split("/").pop() || `image-${i + 1}`,
        }));
        state.sources.push(...sources);
        return sources;
      }

      case "paste_source": {
        const id = `src-${state.nextFrameId++}`;
        const thumb = generateMockImage(200, 200, Date.now(), 1, 1);
        const source = {
          id,
          path: `mock://paste/${id}.png`,
          thumb,
          width: 800,
          height: 600,
          name: "Pasted image",
        };
        state.sources.push(source);
        return source;
      }

      case "frame_to_source": {
        const frameId = args.frame;
        const frame = state.frames.find((f) => f.id === frameId);
        if (!frame) throw new Error("Frame not found");
        const id = `src-${state.nextFrameId++}`;
        const source = {
          id,
          path: frame.path,
          thumb: frame.thumb,
          width: frame.width,
          height: frame.height,
          name: `From ${frameId.slice(0, 8)}`,
          fromFrame: frameId,
        };
        state.sources.push(source);
        return source;
      }

      case "delete_source": {
        state.sources = state.sources.filter((s) => s.id !== args.id);
        return {};
      }

      case "run_tool": {
        const batchId = crypto.randomUUID().slice(0, 8);
        window._emit("batch", { batch: batchId, state: "running" });
        await sleep(1500);
        const frame = {
          id: `${batchId}-0`,
          batch: batchId,
          index: 0,
          path: `mock://tool/${batchId}.png`,
          thumb: generateMockImage(1024, 1024, Date.now(), 1, 1),
          width: 2048,
          height: 2048,
          seed: 0,
          recipe: { tool: args.tool },
          model: "real-esrgan",
          variant: "default",
          elapsed: 1.5,
          saved: null,
          created_at: new Date().toISOString(),
        };
        state.frames.unshift(frame);
        window._emit("frame", frame);
        window._emit("batch", { batch: batchId, state: "done" });
        return { batch: batchId };
      }

      case "save_frame": {
        const frameId = args.frame_id;
        const frame = state.frames.find((f) => f.id === frameId);
        if (!frame) throw new Error("Frame not found");
        const now = new Date();
        const filename = `latent_${now.toISOString().slice(0, 10)}_${now.toISOString().slice(11, 19)}_${frame.model}_${frame.seed}`;
        const savePath = `${state.settings.save_folder}\\${filename}.png`;
        frame.saved = savePath;
        return savePath;
      }

      case "save_frame_as": {
        const frameId = args.frame_id;
        const frame = state.frames.find((f) => f.id === frameId);
        if (!frame) throw new Error("Frame not found");
        frame.saved = `mock://saved/${frameId}.png`;
        return frame.saved;
      }

      case "delete_frame": {
        const frameId = args.frame_id;
        state.frames = state.frames.filter((f) => f.id !== frameId);
        return {};
      }

      case "reveal": {
        return {};
      }

      default:
        throw new Error(`Unknown command: ${cmd}`);
    }
  };

  window.listen = function (event, cb) {
    if (!listeners[event]) listeners[event] = [];
    listeners[event].push(cb);
  };

  window._emit = function (event, payload) {
    const cbs = listeners[event];
    if (cbs) {
      cbs.forEach((cb) => cb({ payload }));
    }
  };

  // ── Mock image generation ──────────────────────────────────────

  function generateMockImage(width, height, seed, step, total) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(width, 256);
    canvas.height = Math.min(height, 256);
    const ctx = canvas.getContext("2d");

    // Seeded pseudo-random noise
    const rng = mulberry32(seed + step * 1000);

    // Gradient background
    const grad = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
    const hue = rng() * 360;
    grad.addColorStop(0, `hsl(${hue}, 40%, ${15 + (step / total) * 20}%)`);
    grad.addColorStop(1, `hsl(${(hue + 40) % 360}, 30%, ${10 + (step / total) * 15}%)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Add some shapes
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      ctx.arc(rng() * canvas.width, rng() * canvas.height, 10 + rng() * 30, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${rng() * 360}, 60%, 50%, ${0.05 + (step / total) * 0.1})`;
      ctx.fill();
    }

    // Step indicator
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "12px monospace";
    ctx.fillText(`${step}/${total}`, 4, 14);

    return canvas.toDataURL("image/png");
  }

  // Simple seeded RNG (Mulberry32)
  function mulberry32(a) {
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ── Helpers ────────────────────────────────────────────────────

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  // ── HW simulation ──────────────────────────────────────────────
  // Simulate VRAM changes when model is loaded
  setInterval(() => {
    if (state.resident?.state === "warm") {
      state.hw.devices[0].vram_used_mb = 10240 + Math.floor(Math.random() * 500);
      window._emit("hw", {
        device: state.hw.devices[0].name,
        vram: {
          used: state.hw.devices[0].vram_used_mb,
          total: state.hw.devices[0].vram_total_mb,
        },
        ram: {
          used: state.hw.ram_used + Math.floor(Math.random() * 100),
          total: state.hw.ram_total,
        },
      });
    }
  }, 1000);

  // Make invoke/listen global for app.js
  window.invoke = window.invoke;
  window.listen = window.listen;
  window._emit = window._emit;
})();
