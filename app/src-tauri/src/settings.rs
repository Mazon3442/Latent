/// Settings management: read/write JSON settings file.
/// Path: %APPDATA%\com.mazon.latent (Windows) or ~/.local/share/com.mazon.latent (Linux)
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub theme: Theme,
    pub save_folder: PathBuf,
    pub save_format: SaveFormat,
    pub filename_pattern: String,
    pub warn_close: bool,
    pub models_folder: PathBuf,
    pub idle_unload_minutes: IdleUnload,
    pub warm_start: bool,
    pub compute_target: ComputeTarget,
    pub hf_token: String,
    pub batch_size: usize,
    pub default_aspect: AspectRatio,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Theme {
    Dark,
    Light,
    System,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SaveFormat {
    Png,
    Webp,
    Jpeg,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum IdleUnload {
    Minutes(u32),
    Never,
}

impl Default for IdleUnload {
    fn default() -> Self {
        IdleUnload::Minutes(10)
    }
}

impl IdleUnload {
    pub fn minutes(&self) -> Option<u32> {
        match self {
            IdleUnload::Minutes(m) => Some(*m),
            IdleUnload::Never => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ComputeTarget {
    Auto,
    Gpu,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AspectRatio {
    R1_1,
    R4_5,
    R3_2,
    R16_9,
    R9_16,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: Theme::Dark,
            save_folder: default_save_folder(),
            save_format: SaveFormat::Png,
            filename_pattern: "latent_{date}_{time}_{model}_{seed}".into(),
            warn_close: true,
            models_folder: default_models_folder(),
            idle_unload_minutes: IdleUnload::default(),
            warm_start: false,
            compute_target: ComputeTarget::Auto,
            hf_token: String::new(),
            batch_size: 1,
            default_aspect: AspectRatio::R1_1,
        }
    }
}

/// Return the settings file path.
pub fn settings_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("com.mazon.latent")
}

fn default_save_folder() -> PathBuf {
    dirs::picture_dir()
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_else(|| PathBuf::from(".")))
        .join("Latent")
}

fn default_models_folder() -> PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("latent")
}

/// Load settings from disk, or return defaults if not found.
pub fn load() -> Settings {
    let path = settings_path();
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(settings) = serde_json::from_str(&content) {
                return settings;
            }
        }
    }
    Settings::default()
}

/// Save settings to disk.
pub fn save(settings: &Settings) -> Result<(), String> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create settings dir: {}", e))?;
    }
    let content = serde_json::to_string_pretty(settings)
        .map_err(|e| format!("Failed to serialize settings: {}", e))?;
    std::fs::write(&path, content).map_err(|e| format!("Failed to write settings: {}", e))?;
    Ok(())
}
