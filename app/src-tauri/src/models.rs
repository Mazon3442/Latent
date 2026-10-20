/// Model catalog: known models, variants, capabilities.
/// This is the static data that the UI uses to render the model picker
/// and the backend uses to validate requests.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub family: String,
    pub desc: String,
    pub license: String,
    pub license_url: String,
    pub commercial: bool,
    pub can: Capabilities,
    pub sizes: Vec<[usize; 2]>,
    pub grid: u8,
    pub options: ModelOptions,
    pub variants: Vec<ModelVariant>,
    pub default_variant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capabilities {
    pub generate: bool,
    pub edit: bool,
    pub inpaint: bool,
    pub multi_ref: bool,
    pub negative: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelOptions {
    pub steps: StepOptions,
    pub guidance: Option<StepOptions>,
    pub sampler: Option<Vec<String>>,
    pub strength: Option<StepOptions>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepOptions {
    pub min: f64,
    pub max: f64,
    pub default: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVariant {
    pub id: String,
    pub label: String,
    pub vram_mb: u64,
    pub ram_mb: u64,
    pub disk_mb: u64,
    pub quantized: bool,
    pub offloaded: bool,
    pub fits: FitVerdict,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    Yes,
    Tight,
    No,
}

impl std::fmt::Display for FitVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitVerdict::Yes => write!(f, "fits"),
            FitVerdict::Tight => write!(f, "tight"),
            FitVerdict::No => write!(f, "doesn't fit"),
        }
    }
}

/// Build the default model catalog.
pub fn catalog() -> Vec<Model> {
    vec![
        z_image_turbo(),
        flux3_klein_comfy(),
        qwen_image_2_1_turbo(),
        sdxl_1_0(),
        real_esrgan(),
        birefnet(),
    ]
}

fn z_image_turbo() -> Model {
    let vram = 13 * 1024; // ~13 GB for 16GB GPU
    Model {
        id: "z-image-turbo".into(),
        name: "Z-Image Turbo".into(),
        family: "z-image".into(),
        desc: "Text-to-image, distilled to 8 steps. Best for quick generation.".into(),
        license: "Apache 2.0".into(),
        license_url: "".into(),
        commercial: true,
        can: Capabilities {
            generate: true,
            edit: false,
            inpaint: false,
            multi_ref: false,
            negative: false,
        },
        sizes: vec![
            [512, 512],
            [768, 768],
            [1024, 576],
            [1024, 640],
            [1152, 640],
            [576, 1024],
            [640, 1024],
            [640, 1152],
            [832, 1216],
            [1216, 832],
        ],
        grid: 16,
        options: ModelOptions {
            steps: StepOptions {
                min: 4.0,
                max: 8.0,
                default: 8.0,
            },
            guidance: Some(StepOptions {
                min: 1.0,
                max: 10.0,
                default: 4.5,
            }),
            sampler: Some(vec![
                "DPM++ 2M".into(),
                "DPM++ SDE".into(),
                "Euler a".into(),
            ]),
            strength: None,
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(),
                label: "BF16".into(),
                vram_mb: vram,
                ram_mb: 2 * 1024,
                disk_mb: 12 * 1024,
                quantized: false,
                offloaded: false,
                fits: if vram <= 16 * 1024 {
                    FitVerdict::Yes
                } else {
                    FitVerdict::Tight
                },
            },
            ModelVariant {
                id: "fp8".into(),
                label: "FP8 (quantized)".into(),
                vram_mb: 8 * 1024,
                ram_mb: 3 * 1024,
                disk_mb: 6 * 1024,
                quantized: true,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
        ],
        default_variant: "bf16".into(),
    }
}

fn flux2_klein_4b() -> Model {
    let vram = 10 * 1024;
    Model {
        id: "flux2-klein-4b".into(),
        name: "FLUX.2 [klein] 4B".into(),
        family: "flux2".into(),
        desc: "Generate and edit images. Supports multiple reference images for style transfer."
            .into(),
        license: "Apache 2.0".into(),
        license_url: "".into(),
        commercial: true,
        can: Capabilities {
            generate: true,
            edit: true,
            inpaint: false,
            multi_ref: true,
            negative: false,
        },
        sizes: vec![
            [1024, 1024],
            [1152, 896],
            [896, 1152],
            [1216, 832],
            [832, 1216],
            [1344, 768],
            [768, 1344],
            [1536, 640],
            [640, 1536],
        ],
        grid: 16,
        options: ModelOptions {
            steps: StepOptions {
                min: 8.0,
                max: 50.0,
                default: 28.0,
            },
            guidance: Some(StepOptions {
                min: 1.0,
                max: 15.0,
                default: 6.0,
            }),
            sampler: Some(vec![
                "Euler".into(),
                "Euler a".into(),
                "DPM++ 2M".into(),
                "DPM++ SDE".into(),
            ]),
            strength: Some(StepOptions {
                min: 0.0,
                max: 1.0,
                default: 0.55,
            }),
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(),
                label: "BF16".into(),
                vram_mb: vram,
                ram_mb: 2 * 1024,
                disk_mb: 8 * 1024,
                quantized: false,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
            ModelVariant {
                id: "nf4".into(),
                label: "NF4 (quantized)".into(),
                vram_mb: 6 * 1024,
                ram_mb: 4 * 1024,
                disk_mb: 3 * 1024,
                quantized: true,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
        ],
        default_variant: "bf16".into(),
    }
}

fn flux3_klein_comfy() -> Model {
    let vram = 9 * 1024;
    Model {
        id: "flux-3-image-klein-comfy".into(),
        name: "FLUX.3 Klein Comfy".into(),
        family: "flux3".into(),
        desc: "FLUX 3 Klein distilled for ComfyUI-style workflows. Generate and edit images with strong prompt following and fast inference."
            .into(),
        license: "Apache 2.0".into(),
        license_url: "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy".into(),
        commercial: true,
        can: Capabilities {
            generate: true,
            edit: true,
            inpaint: true,
            multi_ref: false,
            negative: true,
        },
        sizes: vec![
            [1024, 1024],
            [1152, 896],
            [896, 1152],
            [1216, 832],
            [832, 1216],
            [1344, 768],
            [768, 1344],
            [1536, 640],
            [640, 1536],
        ],
        grid: 16,
        options: ModelOptions {
            steps: StepOptions {
                min: 4.0,
                max: 30.0,
                default: 24.0,
            },
            guidance: Some(StepOptions {
                min: 1.0,
                max: 15.0,
                default: 7.0,
            }),
            sampler: Some(vec![
                "Euler".into(),
                "DPM++ 2M".into(),
                "DPM++ SDE".into(),
            ]),
            strength: Some(StepOptions {
                min: 0.0,
                max: 1.0,
                default: 0.75,
            }),
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(),
                label: "BF16".into(),
                vram_mb: vram,
                ram_mb: 4 * 1024,
                disk_mb: 18 * 1024,
                quantized: false,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
            ModelVariant {
                id: "fp8".into(),
                label: "FP8 (quantized)".into(),
                vram_mb: 7 * 1024,
                ram_mb: 6 * 1024,
                disk_mb: 9 * 1024,
                quantized: true,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
        ],
        default_variant: "bf16".into(),
    }
}

fn qwen_image_2_1_turbo() -> Model {
    let vram = 12 * 1024;
    Model {
        id: "qwen-image-2-1-turbo".into(),
        name: "Qwen Image 2.1 Turbo".into(),
        family: "qwen-image".into(),
        desc: "Qwen's multimodal image generation model. Strong on text rendering, object editing, and inpainting."
            .into(),
        license: "Apache 2.0".into(),
        license_url: "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo".into(),
        commercial: true,
        can: Capabilities {
            generate: true,
            edit: true,
            inpaint: true,
            multi_ref: false,
            negative: true,
        },
        sizes: vec![
            [1024, 1024],
            [1152, 896],
            [896, 1152],
            [1216, 832],
            [832, 1216],
            [1344, 768],
            [768, 1344],
        ],
        grid: 16,
        options: ModelOptions {
            steps: StepOptions {
                min: 8.0,
                max: 50.0,
                default: 30.0,
            },
            guidance: Some(StepOptions {
                min: 1.0,
                max: 12.0,
                default: 7.5,
            }),
            sampler: Some(vec![
                "DPM++ 2M".into(),
                "DPM++ SDE".into(),
                "Euler".into(),
            ]),
            strength: Some(StepOptions {
                min: 0.0,
                max: 1.0,
                default: 0.7,
            }),
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(),
                label: "BF16".into(),
                vram_mb: vram,
                ram_mb: 8 * 1024,
                disk_mb: 24 * 1024,
                quantized: false,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
            ModelVariant {
                id: "fp8".into(),
                label: "FP8 (quantized)".into(),
                vram_mb: 9 * 1024,
                ram_mb: 6 * 1024,
                disk_mb: 12 * 1024,
                quantized: true,
                offloaded: false,
                fits: FitVerdict::Yes,
            },
        ],
        default_variant: "bf16".into(),
    }
}

fn qwen_image_edit() -> Model {
    let vram = 14 * 1024; // quantized + offloaded fits 16GB
    Model {
        id: "qwen-image-edit".into(),
        name: "Qwen-Image-Edit".into(),
        family: "qwen".into(),
        desc: "Strong on text and object editing. Use with a source image.".into(),
        license: "Apache 2.0".into(),
        license_url: "".into(),
        commercial: true,
        can: Capabilities {
            generate: false,
            edit: true,
            inpaint: true,
            multi_ref: false,
            negative: true,
        },
        sizes: vec![
            [1024, 1024],
            [1152, 896],
            [896, 1152],
            [1216, 832],
            [832, 1216],
        ],
        grid: 64,
        options: ModelOptions {
            steps: StepOptions {
                min: 4.0,
                max: 50.0,
                default: 30.0,
            },
            guidance: Some(StepOptions {
                min: 1.0,
                max: 10.0,
                default: 7.0,
            }),
            sampler: Some(vec!["DPM++ 2M".into(), "DPM++ SDE".into()]),
            strength: Some(StepOptions {
                min: 0.0,
                max: 1.0,
                default: 0.7,
            }),
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(),
                label: "BF16".into(),
                vram_mb: 18 * 1024,
                ram_mb: 4 * 1024,
                disk_mb: 38 * 1024,
                quantized: false,
                offloaded: false,
                fits: FitVerdict::No,
            },
            ModelVariant {
                id: "nf4-offload".into(),
                label: "NF4 + offloaded".into(),
                vram_mb: vram,
                ram_mb: 8 * 1024,
                disk_mb: 10 * 1024,
                quantized: true,
                offloaded: true,
                fits: FitVerdict::Tight,
            },
        ],
        default_variant: "nf4-offload".into(),
    }
}

fn sdxl_1_0() -> Model {
    let vram = 8 * 1024;
    Model {
        id: "sdxl-1.0".into(),
        name: "SDXL 1.0".into(),
        family: "sdxl".into(),
        desc: "Open-source diffusion model. Runs on smaller GPUs. Community checkpoints supported (phase 3).".into(),
        license: "CreativeML OpenRAIL++-M".into(),
        license_url: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0".into(),
        commercial: true,
        can: Capabilities { generate: true, edit: false, inpaint: true, multi_ref: false, negative: true },
        sizes: vec![
            [1024, 1024],
            [1152, 896], [896, 1152],
            [1216, 832], [832, 1216],
            [1344, 768], [768, 1344],
            [1536, 640], [640, 1536],
        ],
        grid: 64,
        options: ModelOptions {
            steps: StepOptions { min: 10.0, max: 100.0, default: 30.0 },
            guidance: Some(StepOptions { min: 1.0, max: 15.0, default: 7.5 }),
            sampler: Some(vec!["DPM++ 2M".into(), "DPM++ SDE".into(), "Euler a".into(), "DDIM".into()]),
            strength: Some(StepOptions { min: 0.0, max: 1.0, default: 0.65 }),
        },
        variants: vec![
            ModelVariant {
                id: "bf16".into(), label: "BF16".into(),
                vram_mb: vram, ram_mb: 2 * 1024, disk_mb: (6.5 * 1024.0) as u64,
                quantized: false, offloaded: false,
                fits: FitVerdict::Yes,
            },
            ModelVariant {
                id: "fp16".into(), label: "FP16".into(),
                vram_mb: vram / 2, ram_mb: 3 * 1024, disk_mb: (3.5 * 1024.0) as u64,
                quantized: false, offloaded: false,
                fits: FitVerdict::Yes,
            },
        ],
        default_variant: "fp16".into(),
    }
}

fn real_esrgan() -> Model {
    Model {
        id: "real-esrgan".into(),
        name: "Real-ESRGAN x4plus".into(),
        family: "esrgan".into(),
        desc: "Upscale 2x or 4x. Works on any frame or source.".into(),
        license: "BSD-3-Clause".into(),
        license_url: "".into(),
        commercial: true,
        can: Capabilities {
            generate: false,
            edit: false,
            inpaint: false,
            multi_ref: false,
            negative: false,
        },
        sizes: vec![],
        grid: 1,
        options: ModelOptions {
            steps: StepOptions {
                min: 1.0,
                max: 1.0,
                default: 1.0,
            },
            guidance: None,
            sampler: None,
            strength: None,
        },
        variants: vec![ModelVariant {
            id: "default".into(),
            label: "x4plus".into(),
            vram_mb: 512,
            ram_mb: 1 * 1024,
            disk_mb: 200,
            quantized: false,
            offloaded: false,
            fits: FitVerdict::Yes,
        }],
        default_variant: "default".into(),
    }
}

fn birefnet() -> Model {
    Model {
        id: "birefnet".into(),
        name: "BiRefNet".into(),
        family: "birefnet".into(),
        desc: "Remove background from any frame or source.".into(),
        license: "MIT".into(),
        license_url: "".into(),
        commercial: true,
        can: Capabilities {
            generate: false,
            edit: false,
            inpaint: false,
            multi_ref: false,
            negative: false,
        },
        sizes: vec![],
        grid: 1,
        options: ModelOptions {
            steps: StepOptions {
                min: 1.0,
                max: 1.0,
                default: 1.0,
            },
            guidance: None,
            sampler: None,
            strength: None,
        },
        variants: vec![ModelVariant {
            id: "default".into(),
            label: "default".into(),
            vram_mb: 1024,
            ram_mb: 2 * 1024,
            disk_mb: 220,
            quantized: false,
            offloaded: false,
            fits: FitVerdict::Yes,
        }],
        default_variant: "default".into(),
    }
}

/// Check which models can handle the given capabilities.
pub fn filter_by_capabilities(models: &[Model], caps: &Capabilities) -> Vec<String> {
    models
        .iter()
        .filter(|m| {
            m.can.generate == caps.generate
                && m.can.edit == caps.edit
                && m.can.inpaint == caps.inpaint
                && m.can.multi_ref == caps.multi_ref
        })
        .map(|m| m.id.clone())
        .collect()
}

/// Find the best variant for a model given available VRAM.
pub fn best_variant(model: &Model, available_vram_mb: u64) -> Option<ModelVariant> {
    let mut fits = model
        .variants
        .iter()
        .filter(|v| {
            v.fits == FitVerdict::Yes
                || (v.fits == FitVerdict::Tight && v.vram_mb <= available_vram_mb)
        })
        .collect::<Vec<_>>();
    fits.sort_by_key(|v| v.vram_mb);
    fits.into_iter().next().cloned()
}
