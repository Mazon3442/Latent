//! What Latent can run: one entry per model, the only place model facts live. The installer, the model manager and
//! the interface all read from here.

use serde::Serialize;

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct Can {
    pub generate: bool,
    pub edit: bool,
    pub inpaint: bool,
    pub multi_ref: bool,
    pub negative: bool,
}

#[derive(Serialize, Clone, Copy)]
pub struct Range {
    pub min: f64,
    pub max: f64,
    pub default: f64,
}

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct Options {
    pub steps: Range,
    /// None for distilled models, which run without classifier-free guidance.
    pub guidance: Option<Range>,
}

#[derive(Serialize, Clone, Copy)]
pub struct Variant {
    pub id: &'static str,
    pub label: &'static str,
    /// Peak GB while generating at the default size; replaced by measurements as they come in.
    pub vram: f64,
    pub ram: f64,
    pub disk: f64,
    /// "none": everything on the GPU. "text": the text encoder runs on the CPU, the rest stays on the GPU.
    /// "model": each part moves to the GPU only while it runs. "stream": the big parts wait in RAM and visit the GPU
    /// one block at a time. "disk": the same, waiting on disk instead, so RAM only holds the part being loaded.
    #[serde(skip)]
    pub offload: &'static str,
    /// How the transformer's weights are stored in memory: "bf16", or "fp8" (half the memory; it still computes in
    /// bf16, one layer at a time).
    #[serde(skip)]
    pub weights: &'static str,
}

impl Variant {
    /// Whether this variant moves its weights through the GPU a block at a time: slow, but it needs only a few GB of VRAM.
    pub fn streams(&self) -> bool {
        matches!(self.offload, "stream" | "disk")
    }
}

#[derive(Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct Model {
    pub id: &'static str,
    pub name: &'static str,
    /// Which loader the worker uses.
    pub family: &'static str,
    pub license: &'static str,
    pub license_url: &'static str,
    pub commercial: bool,
    pub can: Can,
    /// Aspect label and the model's native width and height for it.
    pub sizes: &'static [(&'static str, u32, u32)],
    /// Width and height must be multiples of this.
    pub grid: u32,
    pub options: Options,
    pub variants: &'static [Variant],
    /// GB of RAM when the whole model runs on the CPU, in bf16 (2 bytes a parameter, plus about 2.5 GB for the frame).
    /// Measured for SANA-Sprint and FLUX.2 klein on a Ryzen 7 7800X3D; the others are worked out from their parameter
    /// counts, so treat them as upper bounds.
    pub cpu: f64,
    /// The repo is gated: downloading needs a free Hugging Face account that accepted the license, and its token.
    pub account: bool,
    /// Another model this one runs on (its text encoder and VAE): it can only be downloaded and run while that one is installed.
    pub requires: Option<&'static str>,
    /// Shown before the download, and asks for confirmation.
    pub warning: Option<&'static str>,
    #[serde(skip)]
    pub repo: &'static str,
    /// Folder under <models folder>/models.
    #[serde(skip)]
    pub folder: &'static str,
    /// Repo files the app never loads (sample images, PDFs).
    #[serde(skip)]
    pub skip_files: &'static [&'static str],
}

    // Streaming (measured on an RTX 5070 Ti, 1024 x 1024): the transformer and text encoder wait in RAM or on disk and visit
    // the GPU a block at a time, so a few GB of VRAM is enough and every step is several times slower. Disk streaming
    // keeps only the part being loaded in RAM, and keeps a bf16 copy of the weights in the models folder while loaded.
pub const CATALOG: &[Model] = &[Model {
    id: "z-image-turbo",
    name: "Z-Image Turbo",
    family: "zimage",
    license: "Apache 2.0",
    license_url: "https://huggingface.co/Tongyi-MAI/Z-Image-Turbo",
    commercial: true,
    can: Can { generate: true, edit: false, inpaint: false, multi_ref: false, negative: false },
    sizes: &[("1:1", 1024, 1024), ("4:5", 896, 1120), ("3:2", 1248, 832), ("2:3", 832, 1248), ("16:9", 1344, 768), ("9:16", 768, 1344)],
    grid: 16,
    options: Options { steps: Range { min: 4.0, max: 16.0, default: 9.0 }, guidance: None },
    // The repo stores the transformer in fp32; it loads as bf16. Measured on an RTX 5070 Ti at 1024 x 1024, 9 steps:
    // with the text encoder on the CPU (about 4 s per new prompt, then cached) a frame takes about 13 s and nothing
    // moves between frames. Full offloading needs less VRAM but holds 20 GB in RAM and swaps it every frame.
    variants: &[
        Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 14.2, ram: 10.0, disk: 32.9, offload: "text", weights: "bf16" },
        Variant { id: "bf16-offload", label: "BF16, offloaded", vram: 12.3, ram: 22.0, disk: 32.9, offload: "model", weights: "bf16" },
        Variant { id: "bf16-stream", label: "BF16, streamed from RAM (slow)", vram: 2.6, ram: 21.9, disk: 32.9, offload: "stream", weights: "bf16" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 19 GB on disk while loaded)", vram: 2.6, ram: 18.9, disk: 32.9, offload: "disk", weights: "bf16" },
    ],
    cpu: 23.0,
    account: false,
    requires: None,
    warning: None,
    repo: "Tongyi-MAI/Z-Image-Turbo",
    folder: "Z-Image-Turbo",
    skip_files: &["assets/*", "*.pdf"],
}, Model {
    id: "flux3-klein",
    name: "FLUX.3 klein",
    family: "flux3klein",
    license: "FLUX Kommunity License (free under US$5M revenue)",
    license_url: "https://huggingface.co/black-forest-labs/flux-3-image-klein",
    commercial: true,
    can: Can { generate: true, edit: true, inpaint: false, multi_ref: true, negative: false },
    sizes: &[("1:1", 1024, 1024), ("4:5", 896, 1120), ("3:2", 1248, 832), ("2:3", 832, 1248), ("16:9", 1344, 768), ("9:16", 768, 1344)],
    grid: 16,
    options: Options { steps: Range { min: 8.0, max: 32.0, default: 16.0 }, guidance: None },
    // Measured on an RTX 5070 Ti at 1024 x 1024, 16 steps, text encoder on the CPU: 12.9 GB peak, about 17 s per new
    // image (0.8 s per step) and 33 s per edit with one reference (the reference doubles the tokens). The offloaded
    // variant is an estimate, not yet measured.
    variants: &[
        Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 12.9, ram: 11.0, disk: 20.4, offload: "text", weights: "bf16" },
        Variant { id: "bf16-offload", label: "BF16, offloaded", vram: 11.0, ram: 22.0, disk: 20.4, offload: "model", weights: "bf16" },
        Variant { id: "bf16-stream", label: "BF16, streamed from RAM (slow)", vram: 3.8, ram: 21.2, disk: 20.4, offload: "stream", weights: "bf16" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 19 GB on disk while loaded)", vram: 3.8, ram: 11.1, disk: 20.4, offload: "disk", weights: "bf16" },
    ],
    cpu: 22.6,
    account: true,
    requires: None,
    warning: None,
    repo: "black-forest-labs/flux-3-image-klein",
    folder: "FLUX.3-klein",
    // The repo also holds single-file copies and FP8/NVFP4 variants for other tools; Latent loads the diffusers folders.
    skip_files: &["assets/*", "variants/*", "*.jpg", "model.safetensors", "ae.safetensors", "config.json", "manifest.json"],
}, Model {
    id: "qwen-image-2.1-turbo",
    name: "Qwen-Image 2.1 Turbo",
    family: "qwenimage21",
    license: "Qwen research license (non-commercial)",
    license_url: "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo",
    commercial: false,
    can: Can { generate: true, edit: true, inpaint: false, multi_ref: false, negative: false },
    // The pipeline's default output is 1024 px; these are the model card's 2K presets at half size (a quarter of the
    // pixels, so about four times faster).
    sizes: &[("1:1", 1024, 1024), ("4:3", 1200, 896), ("3:4", 896, 1200), ("3:2", 1264, 848), ("2:3", 848, 1264), ("16:9", 1376, 768), ("9:16", 768, 1376)],
    grid: 16,
    // The checkpoint carries its own 8-step schedule; the step count can't be changed.
    options: Options { steps: Range { min: 8.0, max: 8.0, default: 8.0 }, guidance: None },
    // Measured on an RTX 5070 Ti at 1024 x 1024: the 7B transformer stored as FP8 (7.3 GB loaded), the 17.5 GB text
    // encoder on the CPU, VAE tiled. About 17-24 s per new image (7.9 GB peak) and 39 s per edit (10.5 GB peak; the
    // text encoder reads the reference too). Untiled, an edit peaked at 15.9 GB.
    variants: &[
        Variant { id: "fp8-text-cpu", label: "FP8 weights, text encoder on the CPU", vram: 10.5, ram: 20.0, disk: 32.9, offload: "text", weights: "fp8" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 30 GB on disk while loaded)", vram: 1.9, ram: 17.1, disk: 32.9, offload: "disk", weights: "bf16" },
    ],
    cpu: 34.9,
    account: false,
    requires: None,
    warning: None,
    repo: "Qwen/Qwen-Image-2.1-Turbo",
    folder: "Qwen-Image-2.1-Turbo",
    skip_files: &["assets/*", "*.jpg", "*.png", "*.webp", "*.gif"],
}, Model {
    id: "flux2-klein-4b",
    name: "FLUX.2 klein 4B",
    family: "flux2klein",
    license: "Apache 2.0",
    license_url: "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B",
    commercial: true,
    can: Can { generate: true, edit: true, inpaint: false, multi_ref: true, negative: false },
    sizes: &[("1:1", 1024, 1024), ("4:5", 896, 1120), ("3:2", 1248, 832), ("2:3", 832, 1248), ("16:9", 1344, 768), ("9:16", 768, 1344)],
    grid: 16,
    // Step-distilled: the model card runs it in 4 steps at guidance 1.
    options: Options { steps: Range { min: 2.0, max: 8.0, default: 4.0 }, guidance: None },
    // Measured on an RTX 5070 Ti at 1024 x 1024, 4 steps, text encoder on the CPU: BF16 9.8 GB peak, about 12 s per
    // new image and 11 s per edit; FP8 6.2 GB peak at the same speed, the variant 8 GB cards get.
    variants: &[
        Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 9.8, ram: 10.0, disk: 15.9, offload: "text", weights: "bf16" },
        Variant { id: "fp8-text-cpu", label: "FP8 weights, text encoder on the CPU", vram: 6.2, ram: 10.0, disk: 15.9, offload: "text", weights: "fp8" },
        Variant { id: "bf16-stream", label: "BF16, streamed from RAM (slow)", vram: 2.6, ram: 21.3, disk: 15.9, offload: "stream", weights: "bf16" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 15 GB on disk while loaded)", vram: 2.6, ram: 8.2, disk: 15.9, offload: "disk", weights: "bf16" },
    ],
    cpu: 18.6,
    account: false,
    requires: None,
    warning: None,
    repo: "black-forest-labs/FLUX.2-klein-4B",
    folder: "FLUX.2-klein-4B",
    skip_files: &["flux-2-klein-4b.safetensors", "assets/*", "*.jpg", "*.png"],
}, Model {
    id: "sana-sprint-1.6b",
    name: "SANA-Sprint 1.6B",
    family: "sanasprint",
    license: "Apache 2.0, Gemma terms for its text encoder",
    license_url: "https://huggingface.co/Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    commercial: true,
    can: Can { generate: true, edit: false, inpaint: false, multi_ref: false, negative: false },
    // Its autoencoder works in 32 px blocks.
    sizes: &[("1:1", 1024, 1024), ("4:5", 896, 1120), ("3:2", 1248, 832), ("2:3", 832, 1248), ("16:9", 1344, 768), ("9:16", 768, 1344)],
    grid: 32,
    options: Options { steps: Range { min: 1.0, max: 4.0, default: 2.0 }, guidance: None },
    // Measured on an RTX 5070 Ti at 1024 x 1024, 2 steps, text encoder on the CPU: 5.3 GB peak, about 7 s per image.
    variants: &[
        Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 5.3, ram: 7.0, disk: 9.6, offload: "text", weights: "bf16" },
        Variant { id: "bf16-stream", label: "BF16, streamed from RAM (slow)", vram: 2.3, ram: 12.0, disk: 9.6, offload: "stream", weights: "bf16" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 8 GB on disk while loaded)", vram: 2.3, ram: 6.4, disk: 9.6, offload: "disk", weights: "bf16" },
    ],
    cpu: 11.9,
    account: false,
    requires: None,
    warning: None,
    repo: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    folder: "SANA-Sprint-1.6B",
    skip_files: &["assets/*", "*.jpg", "*.png", "*.gif"],
}, Model {
    id: "kromoss",
    name: "Kromoss",
    family: "kromoss",
    license: "Apache 2.0",
    license_url: "https://huggingface.co/lodestones/Kromoss",
    commercial: true,
    can: Can { generate: true, edit: false, inpaint: false, multi_ref: false, negative: false },
    sizes: &[("1:1", 1024, 1024), ("4:5", 896, 1120), ("3:2", 1248, 832), ("2:3", 832, 1248), ("16:9", 1344, 768), ("9:16", 768, 1344)],
    grid: 16,
    // Latent runs it with the card's 8-step DMD2 LoRA at guidance 1 (4 steps also works), not the 50-step, guidance 3
    // full model, which needs a second forward per step.
    options: Options { steps: Range { min: 4.0, max: 16.0, default: 8.0 }, guidance: None },
    // Measured on an RTX 5070 Ti at 1024 x 1024, 8 steps, the 9B text encoder on the CPU (about 3 s for a new prompt, then
    // cached): BF16 13.7 GB peak and about 12 s per image; FP8 9.0 GB peak and about 11 s. The 1.2 GB LoRA is part of both.
    // RAM is the encoder's 17 GB plus headroom, not measured.
    variants: &[
        Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 13.7, ram: 20.0, disk: 28.4, offload: "text", weights: "bf16" },
        Variant { id: "fp8-text-cpu", label: "FP8 weights, text encoder on the CPU", vram: 9.0, ram: 20.0, disk: 28.4, offload: "text", weights: "fp8" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 25 GB on disk while loaded)", vram: 4.3, ram: 16.1, disk: 28.4, offload: "disk", weights: "bf16" },
    ],
    cpu: 29.6,
    account: false,
    requires: None,
    warning: None,
    repo: "lodestones/Kromoss",
    folder: "Kromoss",
    // Latent runs the vendored pipeline in engines/, so none of the repo's code is downloaded. The single-file copies,
    // the ComfyUI text encoder and the second LoRA are for other tools.
    skip_files: &["*.py", "assets/*", "diffusion_models/*", "text_encoders/*", "vae/flux2-vae_small.safetensors", "8-steps-dmd-lora.safetensors", "*.webp"],
}, Model {
    id: "qwen-image-2.1-uncensored",
    name: "Qwen-Image 2.1 Uncensored",
    family: "qwenimage21uc",
    license: "Qwen research license (non-commercial)",
    license_url: "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized",
    commercial: false,
    can: Can { generate: true, edit: true, inpaint: false, multi_ref: false, negative: false },
    sizes: &[("1:1", 1024, 1024), ("4:3", 1200, 896), ("3:4", 896, 1200), ("3:2", 1264, 848), ("2:3", 848, 1264), ("16:9", 1376, 768), ("9:16", 768, 1376)],
    grid: 16,
    // The base Qwen-Image 2.1 weights (not the distilled Turbo): the model card runs 40 steps, without guidance.
    options: Options { steps: Range { min: 20.0, max: 50.0, default: 40.0 }, guidance: None },
    // Only the FP8 transformer (7.1 GB) is downloaded: the text encoder, VAE and processor come from Qwen-Image 2.1 Turbo,
    // which is why that model is required. Measured on an RTX 5070 Ti at 1024 x 1024, 40 steps, text encoder on the CPU:
    // 7.9 GB peak and about 51 s per new image; 10.5 GB and about 107 s per edit.
    variants: &[
        Variant { id: "fp8-text-cpu", label: "FP8 weights, text encoder on the CPU", vram: 10.5, ram: 20.0, disk: 7.1, offload: "text", weights: "fp8" },
        Variant { id: "bf16-disk", label: "BF16, streamed from disk (slowest, keeps 30 GB on disk while loaded)", vram: 1.9, ram: 21.3, disk: 7.1, offload: "disk", weights: "bf16" },
    ],
    cpu: 34.9,
    account: false,
    requires: Some("qwen-image-2.1-turbo"),
    warning: Some("This model has no content filter. It makes adult and sensitive images without refusing, so what it makes depends only on your prompts. Only make images of real people with their permission: you're responsible for what you make. Its license is non-commercial."),
    repo: "abenzerps/Qwen-Image-2.1-Uncensored-Quantized",
    folder: "Qwen-Image-2.1-Uncensored",
    // The repo holds every quantization and the ComfyUI text encoder and VAE; Latent loads one FP8 file.
    skip_files: &["*.gguf", "*NVFP4*", "*MLX*", "*int8_convrot*", "*lora*", "text_encoders/*", "vae/*", "assets/*", "SHA256SUMS"],
}];

pub fn find(id: &str) -> Result<&'static Model, String> {
    CATALOG.iter().find(|m| m.id == id).ok_or_else(|| format!("Latent doesn't know a model called {id}"))
}

/// "yes" with a GB to spare, "tight" when it only just fits, "no" when `need` GB is more than `have` (0: unknown).
fn judge(need: f64, have: f64) -> &'static str {
    if have <= 0.0 || need <= have - 1.0 { "yes" } else if need <= have { "tight" } else { "no" }
}

/// The worse of two verdicts.
fn worse(a: &'static str, b: &'static str) -> &'static str {
    ["no", "tight", "yes"].into_iter().find(|v| *v == a || *v == b).unwrap_or("yes")
}

/// Whether a variant fits a GPU with `vram_total` GB, and the RAM beside it. A unified-memory Mac (a GPU with no VRAM
/// figure of its own) holds the whole model, GPU and CPU parts alike, in RAM, so it has nothing to stream between; and
/// Metal can't store FP8 weights, so FP8 variants don't run there either.
pub fn fits(v: &Variant, vram_total: f64, ram_total: f64) -> &'static str {
    let unified = vram_total <= 0.0;
    if unified && (v.weights == "fp8" || v.streams()) {
        return "no";
    }
    if unified { judge(v.vram + v.ram, ram_total) } else { worse(judge(v.vram, vram_total), judge(v.ram, ram_total)) }
}

impl Model {
    /// Whether the whole model fits in `ram_total` GB of RAM, to run on the CPU.
    pub fn fits_cpu(&self, ram_total: f64) -> &'static str {
        judge(self.cpu, ram_total)
    }

    /// The variant asked for, or else the fastest one that fits this PC (variants are listed fastest first). With
    /// `lightest`, the one that needs the least memory instead, to leave room for other apps. On the CPU there is
    /// one way to run (everything in RAM), so the variants only matter on a GPU.
    pub fn variant(&self, id: Option<&str>, on_gpu: bool, vram_total: f64, ram_total: f64, lightest: bool) -> &'static Variant {
        if let Some(v) = id.and_then(|id| self.variants.iter().find(|v| v.id == id)) {
            return v;
        }
        if !on_gpu {
            return &self.variants[0];
        }
        if lightest {
            // Streaming needs the least VRAM of all, but it is several times slower: it is for PCs with no other way.
            let need = |v: &&Variant| if vram_total <= 0.0 { v.vram + v.ram } else { v.vram };
            let usable = self.variants.iter().filter(|v| !v.streams() && fits(v, vram_total, ram_total) != "no");
            if let Some(v) = usable.min_by(|a, b| need(a).total_cmp(&need(b))) {
                return v;
            }
        }
        ["yes", "tight"]
            .iter()
            .find_map(|want| self.variants.iter().find(|v| fits(v, vram_total, ram_total) == *want))
            .unwrap_or(&self.variants[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_fastest_variant_that_fits() {
        let z = find("z-image-turbo").unwrap();
        assert_eq!(z.variant(None, true, 15.9, 31.0, false).id, "bf16-text-cpu", "16 GB card keeps the transformer resident");
        assert_eq!(z.variant(None, true, 13.5, 31.0, false).id, "bf16-offload", "a 14 GB card falls back to offloading");
        assert_eq!(z.variant(Some("bf16-offload"), true, 15.9, 31.0, false).id, "bf16-offload", "an explicit choice wins");
        let k = find("flux2-klein-4b").unwrap();
        assert_eq!(k.variant(None, true, 15.9, 31.0, false).id, "bf16-text-cpu", "full quality when it fits");
        assert_eq!(k.variant(None, true, 15.9, 31.0, true).id, "fp8-text-cpu", "Prefer less VRAM takes the lightest");
        // A Mac (unified memory, no VRAM figure) holds klein 4B whole in RAM, and can't use FP8 weights.
        assert_eq!(fits(&k.variants[0], 0.0, 16.0), "no", "9.8 + 10 GB is more than a 16 GB Mac has");
        assert_eq!(fits(&k.variants[0], 0.0, 32.0), "yes");
        assert_eq!(fits(&k.variants[1], 0.0, 64.0), "no", "Metal can't store FP8");
        assert_eq!(k.variant(None, true, 0.0, 32.0, true).id, "bf16-text-cpu", "Prefer less VRAM never picks FP8 on a Mac");
    }

    #[test]
    fn a_small_gpu_streams_the_model_as_a_last_resort() {
        let z = find("z-image-turbo").unwrap();
        assert_eq!(z.variant(None, true, 8.0, 32.0, false).id, "bf16-stream", "RAM for the whole model: stream from RAM");
        assert_eq!(z.variant(None, true, 8.0, 20.0, false).id, "bf16-disk", "not enough RAM for it: stream from disk");
        assert_eq!(fits(&z.variants[2], 8.0, 8.0), "no", "even the disk tier needs RAM for the part being loaded");
        assert_eq!(z.variant(None, true, 16.0, 32.0, true).id, "bf16-offload", "Prefer less VRAM doesn't pick the slow tiers");
        assert_eq!(z.variant(None, true, 16.0, 32.0, false).id, "bf16-text-cpu", "a big card never streams");
        let q = find("qwen-image-2.1-turbo").unwrap();
        assert!(q.variants.iter().all(|v| v.id != "bf16-stream"), "Qwen-Image is too big to wait in RAM on most PCs");
        assert_eq!(q.variant(None, true, 8.0, 24.0, false).id, "bf16-disk");
        assert_eq!(fits(&z.variants[2], 0.0, 64.0), "no", "a unified-memory Mac has nothing to stream between");
    }

    #[test]
    fn the_cpu_needs_the_whole_model_in_ram() {
        let sana = find("sana-sprint-1.6b").unwrap();
        let z = find("z-image-turbo").unwrap();
        assert_eq!(sana.fits_cpu(16.0), "yes");
        assert_eq!(sana.fits_cpu(11.0), "no");
        assert_eq!(z.fits_cpu(20.0), "no", "bf16 Z-Image is about 23 GB");
        assert_eq!(z.fits_cpu(32.0), "yes");
        assert_eq!(z.variant(None, false, 0.0, 64.0, true).id, z.variants[0].id, "variants only matter on a GPU");
    }
}
