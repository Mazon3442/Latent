//! Settings, kept as JSON in the app data folder, and the JSON helpers the rest of the app saves with.

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase", default)]
pub struct Settings {
    /// "Dark" | "Light" | "System"
    pub theme: String,
    pub save_dir: String,
    pub warn_unsaved: bool,
    /// Models folder: the Python environment and model weights.
    pub models_dir: String,
    /// "5 minutes" | "10 minutes" | "30 minutes" | "Never"
    pub idle: String,
    /// Where models run: "Automatic" | "GPU" | "CPU".
    pub device: String,
    /// Hugging Face access token, for gated models. Stays on this PC.
    pub hf_token: String,
    /// Frames per batch.
    pub count: u32,
    /// Aspect label, e.g. "1:1".
    pub aspect: String,
    pub last_model: String,
    /// Prompt text size: "Small" | "Medium" | "Large".
    pub prompt_size: String,
    /// What Save writes: "PNG" (keeps the recipe) | "JPEG" (smaller, no recipe).
    pub format: String,
    /// Run each model in its lightest variant that fits, leaving VRAM for other apps (a local LLM, a game).
    pub low_vram: bool,
    /// Load the last model in the background when Latent starts.
    pub preload: bool,
    /// Models shown in the top bar, in order. Empty means every installed model.
    pub pinned: Vec<String>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            theme: "Dark".into(),
            save_dir: default_save_dir(),
            warn_unsaved: true,
            models_dir: default_models_dir(),
            idle: "10 minutes".into(),
            device: "Automatic".into(),
            hf_token: String::new(),
            count: 2,
            aspect: "1:1".into(),
            last_model: String::new(),
            prompt_size: "Medium".into(),
            format: "PNG".into(),
            low_vram: false,
            preload: false,
            pinned: Vec::new(),
        }
    }
}

impl Settings {
    pub fn idle_minutes(&self) -> Option<u64> {
        match self.idle.as_str() {
            "5 minutes" => Some(5),
            "10 minutes" => Some(10),
            "30 minutes" => Some(30),
            _ => None,
        }
    }
}

#[cfg(windows)]
fn default_save_dir() -> String {
    format!("{}\\Pictures\\Latent", std::env::var("USERPROFILE").unwrap_or_else(|_| "C:".into()))
}

/// The XDG Pictures folder if `xdg-user-dir` knows it, else ~/Pictures/Latent.
#[cfg(unix)]
fn default_save_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let pictures = std::process::Command::new("xdg-user-dir")
        .arg("PICTURES")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty() && s != &home)
        .unwrap_or_else(|| format!("{home}/Pictures"));
    format!("{pictures}/Latent")
}

#[cfg(windows)]
fn default_models_dir() -> String {
    format!("{}\\latent", std::env::var("USERPROFILE").unwrap_or_else(|_| "C:".into()))
}

/// $XDG_DATA_HOME/latent/models (usually ~/.local/share/latent/models).
#[cfg(unix)]
fn default_models_dir() -> String {
    let data = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| format!("{}/.local/share", std::env::var("HOME").unwrap_or_default()));
    format!("{data}/latent/models")
}

pub fn load<T: DeserializeOwned + Default>(path: &Path) -> T {
    std::fs::read_to_string(path).ok().and_then(|s| serde_json::from_str(s.trim_start_matches('\u{feff}')).ok()).unwrap_or_default()
}

/// Write via a temp file so a crash never leaves half a JSON file behind.
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let tmp: PathBuf = path.with_extension("tmp");
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&tmp, text).map_err(|e| format!("can't write {}: {e}", tmp.display()))?;
    private(&tmp);
    std::fs::rename(&tmp, path).map_err(|e| format!("can't save {}: {e}", path.display()))
}

/// Readable by this user only: settings hold the Hugging Face token. (On Windows the app data folder is already
/// private to the user.)
pub fn private(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
    #[cfg(not(unix))]
    let _ = path;
}
