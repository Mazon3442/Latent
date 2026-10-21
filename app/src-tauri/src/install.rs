//! Downloads models into the models folder: one Python environment shared by every model (they all run on the same
//! PyTorch and diffusers, and a second PyTorch is about 4 GB), then the model's weights as plain files.
//! Progress goes to the UI as "install" events and to logs/install-<id>.log.

use crate::models::Model;
use crate::worker::{adopt, hidden, venv_python};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

pub const ENV: &str = "diffusion-env";

/// Pinned so an upgrade is a deliberate step: one diffusers build serves every model. FLUX.3 klein's pipeline isn't in
/// a release yet, so this is an exact commit of diffusers main (2026-10-20).
const PACKAGES: &[&str] = &[
    "diffusers @ git+https://github.com/huggingface/diffusers@de988b7ee77c393a63865d265d19e267ef9c7bf5",
    "transformers>=4.57", "accelerate", "safetensors", "pillow", "huggingface_hub",
];
const TORCH: &str = "torch==2.11.*";
/// FLUX.3 klein's text encoder (Qwen3-VL) loads its image and video processors through torchvision. From the same
/// index as torch, so the two are built for each other.
const TORCHVISION: &str = "torchvision==0.26.*";

pub fn python(home: &Path) -> PathBuf {
    venv_python(&home.join(ENV))
}

/// Model files live in <models folder>/models/<name> as plain files (no Hugging Face cache symlinks, which
/// Windows refuses without Developer Mode).
pub fn model_dir(home: &Path, m: &Model) -> PathBuf {
    home.join("models").join(m.folder)
}

fn marker(home: &Path, m: &Model) -> PathBuf {
    model_dir(home, m).join(".latent-installed")
}

pub fn is_installed(home: &Path, m: &Model) -> bool {
    marker(home, m).is_file() && python(home).is_file()
}

fn torch_args(device: &str) -> Vec<&'static str> {
    let index = match device {
        "cuda" => "https://download.pytorch.org/whl/cu128", // includes RTX 50-series (Blackwell)
        "rocm" => "https://download.pytorch.org/whl/rocm7.2",
        "xpu" => "https://download.pytorch.org/whl/xpu",
        _ => "https://download.pytorch.org/whl/cpu",
    };
    vec![TORCH, TORCHVISION, "--index-url", index]
}

fn uv() -> Result<PathBuf, String> {
    let mut probe = Command::new("uv");
    probe.arg("--version").stdout(Stdio::null()).stderr(Stdio::null());
    hidden(&mut probe);
    if probe.status().map(|s| s.success()).unwrap_or(false) {
        return Ok("uv".into());
    }
    let home = std::env::var(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).unwrap_or_default();
    let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    [format!("{home}/.local/bin/{exe}"), format!("{home}/.cargo/bin/{exe}"), format!("{local}\\hermes\\bin\\uv.exe")]
        .into_iter()
        .map(PathBuf::from)
        .find(|p| p.is_file())
        .ok_or_else(|| {
            if cfg!(windows) {
                "Downloading models needs uv. Install it with: winget install astral-sh.uv".into()
            } else {
                "Downloading models needs uv. On Arch: sudo pacman -S uv".into()
            }
        })
}

fn run(mut cmd: Command, log: &Path) -> Result<(), String> {
    let f = std::fs::OpenOptions::new().create(true).append(true).open(log).map_err(|e| e.to_string())?;
    cmd.stdout(f.try_clone().map_err(|e| e.to_string())?).stderr(f).stdin(Stdio::null());
    cmd.env("HF_HUB_DISABLE_SYMLINKS_WARNING", "1").env("PYTHONIOENCODING", "utf-8");
    hidden(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| e.to_string())?;
    adopt(&child);
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        return Ok(());
    }
    let text = std::fs::read_to_string(log).unwrap_or_default();
    let tail: Vec<&str> = text.lines().rev().take(4).collect();
    Err(format!("failed ({status}): {}", tail.into_iter().rev().collect::<Vec<_>>().join(" | ")))
}

pub fn folder_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| match e.file_type() {
        Ok(t) if t.is_dir() => folder_bytes(&e.path()),
        _ => e.metadata().map(|m| m.len()).unwrap_or(0),
    }).sum()
}

/// Sets up the environment for `device` and downloads `m`. Safe to re-run: each step skips work that's done, and an
/// interrupted download resumes.
pub fn install(app: Option<&AppHandle>, home: &Path, logs: &Path, m: &Model, device: &str, hf_token: &str) -> Result<(), String> {
    let uv = uv()?;
    let env = home.join(ENV);
    let py = python(home);
    let dir = model_dir(home, m);
    let log = logs.join(format!("install-{}.log", m.id));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let total_bytes = (m.variants[0].disk * 1e9) as u64;
    let emit = |phase: &str, label: &str, done: Option<u64>| {
        if let Some(app) = app {
            let _ = app.emit("install", json!({ "id": m.id, "phase": phase, "label": label, "done": done, "total": total_bytes }));
        }
        println!("{phase}: {label}");
    };

    if !py.is_file() {
        emit("env", "Creating the Python environment", None);
        let mut c = Command::new(&uv);
        c.args(["venv", "--python", "3.12"]).arg(&env);
        run(c, &log).map_err(|e| format!("Creating the Python environment: {e}"))?;
    }
    emit("torch", "Installing PyTorch", None);
    let mut c = Command::new(&uv);
    c.args(["pip", "install", "--python"]).arg(&py).args(torch_args(device));
    run(c, &log).map_err(|e| format!("Installing PyTorch: {e}"))?;

    emit("packages", "Installing diffusers", None);
    let mut c = Command::new(&uv);
    c.args(["pip", "install", "--python"]).arg(&py).args(PACKAGES);
    run(c, &log).map_err(|e| format!("Installing diffusers: {e}"))?;

    emit("weights", "Downloading the model", Some(folder_bytes(&dir)));
    let skip = m.skip_files.iter().map(|p| format!("'{p}'")).collect::<Vec<_>>().join(", ");
    let mut c = Command::new(&py);
    c.arg("-c").arg(format!(
        "from huggingface_hub import snapshot_download; snapshot_download('{}', local_dir=r'{}', ignore_patterns=[{skip}])",
        m.repo,
        dir.display()
    ));
    if !hf_token.is_empty() {
        c.env("HF_TOKEN", hf_token);
    }
    // The download reports nothing useful on its own, so the folder's size is the progress bar.
    let finished = Arc::new(AtomicBool::new(false));
    if let Some(app) = app.cloned() {
        let (finished, dir, id) = (finished.clone(), dir.clone(), m.id);
        std::thread::spawn(move || {
            while !finished.load(Ordering::SeqCst) {
                let _ = app.emit("install", json!({ "id": id, "phase": "weights", "label": "Downloading the model",
                                                   "done": folder_bytes(&dir), "total": total_bytes }));
                std::thread::sleep(Duration::from_secs(1));
            }
        });
    }
    let fetched = run(c, &log);
    finished.store(true, Ordering::SeqCst);
    fetched.map_err(|e| {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if text.contains("GatedRepoError") || text.contains("401 Client Error") || text.contains("403 Client Error") {
            format!("{} needs a free Hugging Face account that has accepted its license. Settings \u{203a} Hugging Face token explains how.", m.name)
        } else {
            format!("Downloading the model: {e}")
        }
    })?;

    std::fs::write(marker(home, m), device).map_err(|e| e.to_string())?;
    emit("done", "Installed", Some(total_bytes));
    Ok(())
}

/// Deletes a model's files. The shared environment goes too once no model is left.
pub fn remove(home: &Path, m: &Model) -> Result<(), String> {
    let dir = model_dir(home, m);
    if dir.exists() {
        std::fs::remove_dir_all(&dir).map_err(|e| format!("couldn't delete {}: {e}", dir.display()))?;
    }
    if !crate::models::CATALOG.iter().any(|other| is_installed(home, other)) {
        let env = home.join(ENV);
        if env.exists() {
            std::fs::remove_dir_all(&env).map_err(|e| format!("couldn't delete {}: {e}", env.display()))?;
        }
    }
    Ok(())
}
