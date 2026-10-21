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
    /// "model": each part moves to the GPU only while it runs.
    #[serde(skip)]
    pub offload: &'static str,
    /// How the transformer's weights are stored in memory: "bf16", or "fp8" (half the memory; it still computes in
    /// bf16, one layer at a time).
    #[serde(skip)]
    pub weights: &'static str,
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
    #[serde(skip)]
    pub repo: &'static str,
    /// Folder under <models folder>/models.
    #[serde(skip)]
    pub folder: &'static str,
    /// Repo files the app never loads (sample images, PDFs).
    #[serde(skip)]
    pub skip_files: &'static [&'static str],
}

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
    ],
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
    ],
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
    variants: &[Variant { id: "fp8-text-cpu", label: "FP8 weights, text encoder on the CPU", vram: 10.5, ram: 20.0, disk: 32.9, offload: "text", weights: "fp8" }],
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
    ],
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
    variants: &[Variant { id: "bf16-text-cpu", label: "BF16, text encoder on the CPU", vram: 5.3, ram: 7.0, disk: 9.6, offload: "text", weights: "bf16" }],
    repo: "Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
    folder: "SANA-Sprint-1.6B",
    skip_files: &["assets/*", "*.jpg", "*.png", "*.gif"],
}];

pub fn find(id: &str) -> Result<&'static Model, String> {
    CATALOG.iter().find(|m| m.id == id).ok_or_else(|| format!("Latent doesn't know a model called {id}"))
}

/// Whether a variant fits a PC with `have` GB of VRAM (on a GPU) or RAM (on the CPU).
pub fn fits(v: &Variant, on_gpu: bool, vram_total: f64, ram_total: f64) -> &'static str {
    let (need, have) = if on_gpu { (v.vram, vram_total) } else { (v.ram, ram_total) };
    if have <= 0.0 || need <= have - 1.0 { "yes" } else if need <= have { "tight" } else { "no" }
}

impl Model {
    /// The variant asked for, or else the fastest one that fits this PC (variants are listed fastest first).
    pub fn variant(&self, id: Option<&str>, on_gpu: bool, vram_total: f64, ram_total: f64) -> &'static Variant {
        if let Some(v) = id.and_then(|id| self.variants.iter().find(|v| v.id == id)) {
            return v;
        }
        ["yes", "tight"]
            .iter()
            .find_map(|want| self.variants.iter().find(|v| fits(v, on_gpu, vram_total, ram_total) == *want))
            .unwrap_or(&self.variants[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_fastest_variant_that_fits() {
        let z = find("z-image-turbo").unwrap();
        assert_eq!(z.variant(None, true, 15.9, 31.0).id, "bf16-text-cpu", "16 GB card keeps the transformer resident");
        assert_eq!(z.variant(None, true, 13.5, 31.0).id, "bf16-offload", "a 14 GB card falls back to offloading");
        assert_eq!(z.variant(Some("bf16-offload"), true, 15.9, 31.0).id, "bf16-offload", "an explicit choice wins");
    }
}
