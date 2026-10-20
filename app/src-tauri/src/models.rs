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
    /// "none": everything on the GPU. "model": each part moves to the GPU only while it runs.
    #[serde(skip)]
    pub offload: &'static str,
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
    // The repo stores the transformer in fp32; it loads as bf16. With offloading, the text encoder and the
    // transformer take turns on the GPU, which is what lets a 16 GB card run it.
    variants: &[Variant { id: "bf16-offload", label: "BF16, offloaded", vram: 13.0, ram: 22.0, disk: 32.9, offload: "model" }],
    repo: "Tongyi-MAI/Z-Image-Turbo",
    folder: "Z-Image-Turbo",
    skip_files: &["assets/*", "*.pdf"],
}];

pub fn find(id: &str) -> Result<&'static Model, String> {
    CATALOG.iter().find(|m| m.id == id).ok_or_else(|| format!("Latent doesn't know a model called {id}"))
}

impl Model {
    pub fn variant(&self, id: Option<&str>) -> &'static Variant {
        id.and_then(|id| self.variants.iter().find(|v| v.id == id)).unwrap_or(&self.variants[0])
    }
}
