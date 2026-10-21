//! Caelestia shell integration: its live Material You color scheme, so Latent matches the desktop.
//! The scheme lives in $XDG_STATE_HOME/caelestia/scheme.json and changes with the wallpaper.

use serde_json::{json, Value};
use std::path::PathBuf;
use std::time::SystemTime;

pub fn scheme_path() -> PathBuf {
    let state = std::env::var("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".local").join("state"));
    state.join("caelestia").join("scheme.json")
}

/// {"mode": "dark"|"light", "colors": {"surface": "#0a0f0f", ...}} or None if Caelestia isn't installed.
pub fn read() -> Option<Value> {
    let text = std::fs::read_to_string(scheme_path()).ok()?;
    let scheme: Value = serde_json::from_str(&text).ok()?;
    let colors: serde_json::Map<String, Value> = scheme["colours"]
        .as_object()?
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|hex| (k.clone(), Value::from(format!("#{}", hex.trim_start_matches('#'))))))
        .collect();
    Some(json!({ "mode": scheme["mode"].as_str().unwrap_or("dark"), "name": scheme["name"], "colors": colors }))
}

pub fn modified() -> Option<SystemTime> {
    std::fs::metadata(scheme_path()).and_then(|m| m.modified()).ok()
}

/// Whether the scheme file exists right now, so a deletion can be told apart from a write in progress.
pub fn present() -> bool {
    scheme_path().is_file()
}
