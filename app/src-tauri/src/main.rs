// Latent — Tauri application entry point
// Ties hw/models/settings/worker/frame_cache together,
// implements all Tauri commands and event forwarding.

mod hw;
mod models;
mod settings;
mod frame_cache;
mod worker;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use std::io::BufRead as _;
use tokio::sync::Mutex;
use tauri::Emitter;
use tauri::Manager;
use worker::{set_event_tx, WorkerCommand};

/// Internal state shared across Tauri commands.
struct AppState {
    settings: Arc<Mutex<settings::Settings>>,
    hw_info: hw::HwInfo,
    worker: Arc<Mutex<Option<worker::WorkerHandle>>>,
    frame_index: Arc<Mutex<frame_cache::FrameIndex>>,
    sources: Arc<Mutex<Vec<frame_cache::Source>>>,
    /// Track pending frames per batch: batch_id -> (remaining_count, model, variant)
    pending_batches: Arc<Mutex<HashMap<String, (usize, String, String)>>>,
    /// Timestamp of last user activity (for idle unload).
    last_activity: Arc<Mutex<Instant>>,
}

// ── Bootstrap ─────────────────────────────────────────────────────

#[tauri::command]
async fn bootstrap(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    let s = state.settings.clone();
    let hw = state.hw_info.clone();

    // Lock settings to access models_folder
    let settings_ref = s.lock().await;

    // Enrich model catalog with install state from disk
    let catalog = models::catalog();
    let installed_ids: Vec<String> = catalog
        .iter()
        .filter(|m| model_is_installed(&settings_ref.models_folder, m))
        .map(|m| m.id.clone())
        .collect();

    let models_json: Vec<serde_json::Value> = catalog
        .iter()
        .map(|m| model_to_json(m, &installed_ids))
        .collect();

    // Gather sources from import dir
    let sources = state.sources.lock().await.clone();
    let sources_json: Vec<serde_json::Value> = sources
        .iter()
        .map(|src| source_to_json(src))
        .collect();

    // Frame list
    let index = state.frame_index.lock().await;
    let frames_json = frames_to_json(&index.frames);
    let batches = frame_cache::get_batches(&index);
    let unsaved = frame_cache::unsaved_count(&index);

    // Resident model info
    let resident = {
        let w = state.worker.blocking_lock();
        match w.as_ref() {
            Some(w) => serde_json::json!({
                "model": w.model(),
                "variant": w.variant(),
                "state": format!("{}", w.state()),
            }),
            None => serde_json::json!({}),
        }
    };

    Ok(serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "settings": serde_json::to_value(&*settings_ref).unwrap(),
        "hw": serde_json::to_value(&hw).unwrap(),
        "models": models_json,
        "sources": sources_json,
        "frames": frames_json,
        "batches": serde_json::to_value(batches.iter().map(|(k, v)| {
            serde_json::json!({
                "batch": k,
                "frames": serde_json::to_value(v.iter().map(|f| {
                    serde_json::json!({
                        "id": f.id,
                        "batch": f.batch,
                        "index": f.index,
                        "path": f.path.to_string_lossy().to_string(),
                        "thumb": f.thumb.to_string_lossy().to_string(),
                        "width": f.width,
                        "height": f.height,
                        "seed": f.seed,
                        "recipe": serde_json::to_value(&f.recipe).unwrap_or_default(),
                        "model": f.model,
                        "variant": f.variant,
                        "elapsed": f.elapsed,
                        "saved": f.saved.clone(),
                        "created_at": f.created_at.to_rfc3339(),
                    })
                }).collect::<Vec<_>>()).unwrap_or_default(),
            })
        }).collect::<Vec<_>>()).unwrap_or_default(),
        "unsaved": unsaved,
        "resident": resident,
    }))
}

fn model_is_installed(models_folder: &PathBuf, model: &models::Model) -> bool {
    let model_dir = models_folder.join(&model.id);
    model_dir.exists()
}

fn model_to_json(m: &models::Model, installed_ids: &[String]) -> serde_json::Value {
    let mut json = serde_json::to_value(m).unwrap_or_default();
    if let Some(obj) = json.as_object_mut() {
        obj.insert(
            "installed".to_string(),
            serde_json::Value::Bool(installed_ids.contains(&m.id)),
        );
    }
    json
}

fn source_to_json(src: &frame_cache::Source) -> serde_json::Value {
    serde_json::json!({
        "id": src.id,
        "path": src.path.to_string_lossy().to_string(),
        "thumb": src.thumb.to_string_lossy().to_string(),
        "width": src.width,
        "height": src.height,
        "name": src.name,
    })
}

fn frames_to_json(frames: &[frame_cache::Frame]) -> Vec<serde_json::Value> {
    frames
        .iter()
        .map(|f| {
            serde_json::json!({
                "id": f.id,
                "batch": f.batch,
                "index": f.index,
                "path": f.path.to_string_lossy().to_string(),
                "thumb": f.thumb.to_string_lossy().to_string(),
                "width": f.width,
                "height": f.height,
                "seed": f.seed,
                "recipe": serde_json::to_value(&f.recipe).unwrap_or_default(),
                "model": f.model,
                "variant": f.variant,
                "elapsed": f.elapsed,
                "saved": f.saved.clone(),
                "created_at": f.created_at.to_rfc3339(),
            })
        })
        .collect()
}

fn batches_to_json(batches: &[(String, Vec<frame_cache::Frame>)]) -> Vec<serde_json::Value> {
    batches
        .iter()
        .map(|(batch_id, frames)| {
            serde_json::json!({
                "batch": batch_id,
                "frames": frames_to_json(frames),
            })
        })
        .collect()
}

// ── Settings ──────────────────────────────────────────────────────

async fn save_settings_clone(s: &settings::Settings) -> serde_json::Value {
    serde_json::to_value(s).unwrap_or_default()
}

#[tauri::command]
async fn save_settings(
    state: tauri::State<'_, AppState>,
    settings: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let s = serde_json::from_value::<settings::Settings>(settings)
        .map_err(|e| format!("Invalid settings: {}", e))?;

    {
        let mut st = state.settings.lock().await;
        settings::save(&s).map_err(|e| e.to_string())?;
        *st = s.clone();
        drop(st);
    }

    Ok(serde_json::to_value(&s).unwrap_or_default())
}

// ── Models ────────────────────────────────────────────────────────

#[tauri::command]
async fn models(
    state: tauri::State<'_, AppState>,
) -> Result<Vec<serde_json::Value>, String> {
    let catalog = models::catalog();
    let installed_ids: Vec<String> = catalog
        .iter()
        .filter(|m| model_is_installed(&state.settings.blocking_lock().models_folder, m))
        .map(|m| m.id.clone())
        .collect();

    Ok(catalog
        .iter()
        .map(|m| model_to_json(m, &installed_ids))
        .collect())
}

#[tauri::command]
async fn install_model(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    model_id: String,
    variant: String,
) -> Result<serde_json::Value, String> {
    let catalog = models::catalog();
    let model = catalog
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;

    let variant_obj = model
        .variants
        .iter()
        .find(|v| v.id == variant)
        .ok_or_else(|| format!("Unknown variant: {}", variant))?;

    // Emit install events via Tauri
    let phases: Vec<(&str, Option<u64>, Option<u32>, Option<&str>)> = vec![
        ("env", None, None, None),
        ("torch", None, None, None),
        ("weights", Some(variant_obj.disk_mb), Some(100), None),
        ("verify", None, None, None),
        ("done", None, Some(100), None),
    ];

    for (phase, total, done, message) in phases {
        if let Err(e) = handle.emit("install", serde_json::json!({
            "id": model_id,
            "phase": phase,
            "total": total,
            "done": done,
            "message": message,
        })) {
            log::warn!("Failed to emit install event: {}", e);
        }
        // Small delay to simulate work (would be actual download in phase 1)
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    Ok(model_to_json(model, &[model_id.clone()]).into())
}

#[tauri::command]
async fn remove_model(
    state: tauri::State<'_, AppState>,
    model_id: String,
) -> Result<serde_json::Value, String> {
    let catalog = models::catalog();
    let model = catalog
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;

    // Unload if currently resident
    {
        let mut worker_guard = state.worker.lock().await;
        if let Some(ref worker) = *worker_guard {
            if worker.model() == &model_id {
                worker.stop().ok();
                *worker_guard = None;
            }
        }
    }

    // Remove model folder
    let models_folder = &state.settings.lock().await.models_folder;
    let model_dir = models_folder.join(&model_id);
    if model_dir.exists() {
        std::fs::remove_dir_all(&model_dir).ok();
    }

    Ok(model_to_json(model, &[]).into())
}

// ── Engine Lifecycle ──────────────────────────────────────────────

#[tauri::command]
async fn preload(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    model: serde_json::Value,
    variant: serde_json::Value,
) -> Result<(), String> {
    let model_id: String = model.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let variant_id: String = variant.as_str().unwrap_or("").to_string();

    let catalog = models::catalog();
    let model_def = catalog
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Unknown model: {}", model_id))?;

    // Pick variant
    let variant_obj = model_def
        .variants
        .iter()
        .find(|v| v.id == variant_id)
        .or_else(|| model_def.variants.first())
        .ok_or_else(|| "No variants available".to_string())?;

    let s = state.settings.lock().await;
    let compute = match s.compute_target {
        settings::ComputeTarget::Auto => "auto",
        settings::ComputeTarget::Gpu => "gpu",
        settings::ComputeTarget::Cpu => "cpu",
    };

    // Emit engine state: loading
    if let Err(e) = handle.emit("engine", serde_json::json!({
        "model": model_id,
        "variant": variant_id,
        "state": "loading",
        "elapsed": 0.0,
    })) {
        log::warn!("Failed to emit engine event: {}", e);
    }

    // Start worker
    let start = Instant::now();
    let worker = worker::WorkerHandle::start(
        &model_id,
        &variant_id,
        s.models_folder.to_str().ok_or("Invalid models path")?,
        compute,
    )
    .map_err(|e| format!("Failed to preload model: {}", e))?;

    *state.worker.lock().await = Some(worker);
    *state.last_activity.lock().await = Instant::now();

    // Wait for ready event (up to 30s)
    tokio::time::timeout(std::time::Duration::from_secs(30), async {
        // The ready event flows through the global event channel → Tauri emit
        // We just need to wait a moment for it to arrive
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    })
    .await
    .ok();

    let elapsed = start.elapsed().as_secs_f64();
    if let Err(e) = handle.emit("engine", serde_json::json!({
        "model": model_id,
        "variant": variant_id,
        "state": "warm",
        "elapsed": elapsed,
    })) {
        log::warn!("Failed to emit engine event: {}", e);
    }

    Ok(())
}

#[tauri::command]
async fn unload(
    state: tauri::State<'_, AppState>,
    handle: tauri::AppHandle,
) -> Result<(), String> {
    if let Err(e) = handle.emit("engine", serde_json::json!({
        "state": "unloading",
    })) {
        log::warn!("Failed to emit engine event: {}", e);
    }

    if let Some(worker) = state.worker.lock().await.take() {
        worker.stop().ok();
    }

    Ok(())
}

// ── Generate ──────────────────────────────────────────────────────

#[tauri::command]
async fn generate(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    request: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let gen_req: frame_cache::GenRequest =
        serde_json::from_value(request).map_err(|e| format!("Invalid request: {}", e))?;

    let worker = state.worker.lock().await;
    let worker = worker.as_ref().ok_or("No model loaded — preload first")?;

    let batch_id = uuid::Uuid::new_v4().to_string();
    let count = gen_req.count.max(1).min(8);
    let variant = gen_req
        .variant
        .clone()
        .unwrap_or_else(|| "default".into());
    let model_id = gen_req.model.clone();

    // Register pending frames
    {
        let mut batches = state.pending_batches.lock().await;
        batches.insert(
            batch_id.clone(),
            (count, model_id.clone(), variant.clone()),
        );
    }

    // Update last activity
    *state.last_activity.lock().await = Instant::now();

    // Send generate commands for each frame
    for i in 0..count {
        let seed = gen_req
            .seed
            .map(|s| s + i as i64)
            .unwrap_or_else(|| rand::random::<u64>() as i64 % (i64::MAX - 1) + 1);

        let cache_dir = frame_cache::cache_dir();
        let output_path = cache_dir.join(format!("{}.{}.png", batch_id, i));

        let cmd = WorkerCommand::Generate {
            id: format!("{}-{}", batch_id, i),
            prompt: gen_req.prompt.clone(),
            sources: gen_req.sources.clone(),
            mask: gen_req.mask.clone(),
            width: gen_req.width,
            height: gen_req.height,
            seed,
            options: serde_json::Value::Object(serde_json::Map::from_iter(
                gen_req.options.clone().into_iter(),
            )),
            output: output_path.to_string_lossy().to_string(),
        };

        let rx = tokio::sync::mpsc::unbounded_channel().0;
        worker.request_stream(cmd, rx)?;
    }

    Ok(serde_json::json!({"batch": batch_id}))
}

#[tauri::command]
async fn cancel(
    state: tauri::State<'_, AppState>,
    handle: tauri::AppHandle,
    batch_id: String,
) -> Result<(), String> {
    // Cancel all frames in the batch
    let worker = state.worker.lock().await;
    if let Some(worker) = worker.as_ref() {
        // Send cancel for each frame in the batch
        for i in 0..20 {
            let cancel_id = format!("{}-{}", batch_id, i);
            worker.cancel(&cancel_id).ok();
        }
    }

    // Emit cancelled batch
    if let Err(e) = handle.emit(
        "batch",
        serde_json::json!({
            "batch": batch_id,
            "state": "cancelled",
        }),
    ) {
        log::warn!("emit failed: {}", e);
    }

    // Remove from pending
    state.pending_batches.lock().await.remove(&batch_id);

    Ok(())
}

// ── Sources ───────────────────────────────────────────────────────

#[tauri::command]
async fn import_sources(
    state: tauri::State<'_, AppState>,
    paths: Option<Vec<String>>,
) -> Result<Vec<serde_json::Value>, String> {
    let mut sources = state.sources.lock().await;
    let paths = paths.unwrap_or_default();

    for path_str in paths {
        let path = PathBuf::from(&path_str);
        if !path.exists() {
            continue;
        }

        // Get dimensions
        let (width, height) = image_dimensions(&path).unwrap_or((0, 0));

        // Create thumb
        let cache_dir = frame_cache::cache_dir();
        let id = uuid::Uuid::new_v4().to_string();
        let thumb_path = cache_dir.join(format!("{}.thumb.png", id));
        let saved_path = frame_cache::Source::import_source(&path, &thumb_path);

        if let Ok(saved) = saved_path {
            sources.push(frame_cache::Source {
                id: id.clone(),
                path: saved,
                thumb: thumb_path,
                width,
                height,
                name: path
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_default(),
                from_frame: None,
            });
        }
    }

    Ok(sources.iter().map(source_to_json).collect())
}

#[tauri::command]
async fn paste_source(
    state: tauri::State<'_, AppState>,
    png_base64: String,
) -> Result<serde_json::Value, String> {
    let bytes = base64_decode(&png_base64).map_err(|e| format!("Invalid base64: {}", e))?;
    let cache_dir = frame_cache::cache_dir();
    let id = uuid::Uuid::new_v4().to_string();
    let thumb_path = cache_dir.join(format!("{}.thumb.png", id));
    let saved_path = cache_dir.join(format!("{}.paste.png", id));

    // Save PNG
    std::fs::write(&saved_path, &bytes).map_err(|e| e.to_string())?;

    // Create thumbnail
    if let Ok(img) = image::load_from_memory(&bytes) {
        let thumb = image::imageops::thumbnail(&img, 100, 100);
        thumb.save(&thumb_path).ok();
    }

    let source = frame_cache::Source {
        id: id.clone(),
        path: saved_path,
        thumb: thumb_path,
        width: 0,
        height: 0,
        name: "Pasted image".into(),
        from_frame: None,
    };

    state.sources.lock().await.push(source);

    Ok(source_to_json(&state.sources.lock().await.last().unwrap()).into())
}

#[tauri::command]
async fn frame_to_source(
    state: tauri::State<'_, AppState>,
    frame_id: String,
) -> Result<serde_json::Value, String> {
    let index = state.frame_index.lock().await;
    let frame = index
        .frames
        .iter()
        .find(|f| f.id == frame_id)
        .ok_or("Frame not found")?;

    let cache_dir = frame_cache::cache_dir();
    let id = uuid::Uuid::new_v4().to_string();
    let thumb_path = cache_dir.join(format!("{}.src.thumb.png", id));

    let source = frame_cache::Source {
        id: id.clone(),
        path: frame.path.clone(),
        thumb: thumb_path,
        width: frame.width,
        height: frame.height,
        name: format!("From frame {}", frame_id[..8].to_string()),
        from_frame: Some(frame.id.clone()),
    };

    state.sources.lock().await.push(source.clone());

    Ok(source_to_json(&source).into())
}

#[tauri::command]
async fn delete_source(
    state: tauri::State<'_, AppState>,
    id: String,
) -> Result<(), String> {
    state.sources.lock().await.retain(|s| s.id != id);
    Ok(())
}

// ── Tools ─────────────────────────────────────────────────────────

#[tauri::command]
async fn run_tool(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    tool: String,
    input: serde_json::Value,
    scale: Option<f64>,
) -> Result<serde_json::Value, String> {
    let batch_id = uuid::Uuid::new_v4().to_string();

    // Determine which tool model to use
    let tool_model = match tool.as_str() {
        "upscale" => "real-esrgan",
        "cutout" => "birefnet",
        _ => return Err(format!("Unknown tool: {}", tool)),
    };

    let cache_dir = frame_cache::cache_dir();
    let output_path = cache_dir.join(format!("{}.{}.png", batch_id, 0));

    // For tools, we just signal the batch started
    let _ = handle.emit(
        "batch",
        serde_json::json!({
            "batch": batch_id,
            "state": "running",
        }),
    );

    // In phase 1, tools run through the worker or as separate processes.
    // For now, return immediately with a placeholder batch.
    // The actual tool execution would happen here.

    Ok(serde_json::json!({
        "batch": batch_id,
        "tool": tool,
    }))
}

// ── Frame Operations ──────────────────────────────────────────────

#[tauri::command]
async fn save_frame(
    state: tauri::State<'_, AppState>,
    frame_id: String,
) -> Result<String, String> {
    let mut index = state.frame_index.lock().await;

    // Find the frame
    let frame = index
        .frames
        .iter_mut()
        .find(|f| f.id == frame_id)
        .ok_or("Frame not found")?;

    let s = state.settings.lock().await;
    let save_folder = &s.save_folder;

    // Generate filename from pattern
    let now = chrono::Utc::now();
    let filename = s
        .filename_pattern
        .replace("{date}", &now.format("%Y%m%d").to_string())
        .replace("{time}", &now.format("%H%M%S").to_string())
        .replace("{model}", &frame.model)
        .replace("{seed}", &frame.seed.to_string())
        .replace("{index}", &frame.index.to_string());

    let extension = match s.save_format {
        settings::SaveFormat::Png => "png",
        settings::SaveFormat::Webp => "webp",
        settings::SaveFormat::Jpeg => "jpg",
    };

    let save_path = save_folder.join(format!("{}.{}", filename, extension));

    // Copy the frame to the save folder
    if let Ok(_) = std::fs::copy(&frame.path, &save_path) {
        frame.saved = Some(save_path.to_string_lossy().to_string());
        frame_cache::save_index(&index).ok();
        return Ok(save_path.to_string_lossy().to_string());
    }

    Err("Failed to save frame".to_string())
}

#[tauri::command]
async fn save_frame_as(
    handle: tauri::AppHandle,
    state: tauri::State<'_, AppState>,
    frame_id: String,
) -> Result<Option<String>, String> {
    // Open save dialog via shell
    let frame = {
        let index = state.frame_index.lock().await;
        index
            .frames
            .iter()
            .find(|f| f.id == frame_id)
            .cloned()
            .ok_or("Frame not found")?
    };

    use tauri_plugin_shell::ShellExt;
    let result = handle
        .file_dialog(
            Some(tauri_plugin_shell::shell::FileDialogFilter::new("Images", &["png", "jpg", "jpeg"])),
            Some("Save frame as"),
        )
        .map_err(|e| e.to_string())?;

    if let Some(path) = result.as_ref() {
        let mut index = state.frame_index.lock().await;
        if let Some(f) = index.frames.iter_mut().find(|f| f.id == frame_id) {
            std::fs::copy(&f.path, &path).ok();
            f.saved = Some(path.to_string_lossy().to_string());
            frame_cache::save_index(&index).ok();
            return Ok(Some(path.to_string_lossy().to_string()));
        }
    }

    Ok(None)
}

#[tauri::command]
async fn delete_frame(
    state: tauri::State<'_, AppState>,
    frame_id: String,
) -> Result<(), String> {
    let mut index = state.frame_index.lock().await;
    frame_cache::remove_frame(&mut index, &frame_id)?;

    // Remove pending batch if empty
    state
        .pending_batches
        .lock()
        .await
        .retain(|_, (_, _, _)| true);

    Ok(())
}

#[tauri::command]
async fn reveal(path: String) -> Result<(), String> {
    let p = std::path::Path::new(&path);
    if let Some(parent) = p.parent() {
        std::process::Command::new("explorer")
            .arg(parent)
            .spawn()
            .ok();
    }

    // Fallback: just open the containing directory
    let p = std::path::Path::new(&path);
    if let Some(parent) = p.parent() {
        #[cfg(windows)]
        {
            std::process::Command::new("explorer")
                .arg("/select,")
                .arg(p)
                .spawn()
                .ok();
        }
        #[cfg(target_os = "macos")]
        {
            std::process::Command::new("open")
                .arg("-R")
                .arg(p)
                .spawn()
                .ok();
        }
        #[cfg(target_os = "linux")]
        {
            std::process::Command::new("xdg-open")
                .arg(parent)
                .spawn()
                .ok();
        }
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────

fn image_dimensions(path: &std::path::Path) -> Result<(usize, usize), String> {
    let img = image::open(path).map_err(|e| e.to_string())?;
    Ok((img.width() as usize, img.height() as usize))
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(s)
        .map_err(|e| e.to_string())
}

// ── Entry Point ───────────────────────────────────────────────────

fn main() {
    // Initialize logging
    env_logger::init();

    // Detect hardware
    let hw_info = hw::detect();
    log::info!("Hardware detected: {:?}", hw_info);

    // Load settings
    let settings = settings::load();
    log::info!("Settings loaded from {:?}", settings::settings_path());

    // Load frame index
    let frame_index = frame_cache::load_index();
    log::info!("Frame index loaded: {} frames", frame_index.frames.len());

    // Create event channel
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    set_event_tx(event_tx);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| {
            let handle = app.handle().clone();

            // Spawn event forwarding task — reads from worker event channel,
            // emits to Tauri UI via handle.emit().
            let fwd_event_rx = event_rx;
            tauri::async_runtime::spawn(async move {
                while let Some(event_value) = fwd_event_rx.recv().await {
                    // Determine event type and emit with the correct name
                    if let Some(event_map) = event_value.as_object() {
                        if let Some(event_name) = event_map.get("event").and_then(|e| e.as_str()) {
                            let payload = event_value.clone();
                            match event_name {
                                "ready" => {
                                    let _ = handle.emit("engine", payload);
                                }
                                "progress" => {
                                    let _ = handle.emit("progress", payload);
                                }
                                "done" => {
                                    let _ = handle.emit("frame", payload);
                                    // Also emit batch completion if all frames done
                                    let _ = handle.emit("batch", serde_json::json!({
                                        "state": "done",
                                        "frame_id": event_map.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                                    }));
                                }
                                "cancelled" => {
                                    let _ = handle.emit("batch", serde_json::json!({
                                        "state": "cancelled",
                                        "frame_id": event_map.get("id").and_then(|i| i.as_str()).unwrap_or(""),
                                    }));
                                }
                                "error" => {
                                    let _ = handle.emit("batch", serde_json::json!({
                                        "state": "failed",
                                        "kind": event_map.get("kind").and_then(|k| k.as_str()).unwrap_or("unknown"),
                                        "message": event_map.get("message").and_then(|m| m.as_str()).unwrap_or(""),
                                    }));
                                }
                                _ => {}
                            }
                        }
                    }
                }
            });

            // Store state
            let app_state = AppState {
                settings: Arc::new(Mutex::new(settings)),
                hw_info,
                worker: Arc::new(Mutex::new(None)),
                frame_index: Arc::new(Mutex::new(frame_index)),
                sources: Arc::new(Mutex::new(Vec::new())),
                pending_batches: Arc::new(Mutex::new(HashMap::new())),
                last_activity: Arc::new(Mutex::new(Instant::now())),
            };

            tokio::runtime::Handle::current().block_on(async {
                app.manage(app_state);
            });

            // Warm start if configured
            if settings.warm_start {
                let catalog = models::catalog();
                if let Some(model) = catalog.first() {
                    let default_variant = &model.default_variant;
                    handle.emit("engine", serde_json::json!({
                        "model": model.id,
                        "variant": default_variant,
                        "state": "loading",
                    })).ok();

                    // Defer preload until after the app is ready
                    let h = handle.clone();
                    tauri::async_runtime::spawn(async move {
                        // Try to preload the default model
                        let state: tauri::State<AppState> = h.state();
                        let s = state.settings.lock().await;
                        let compute = match s.compute_target {
                            settings::ComputeTarget::Auto => "auto",
                            settings::ComputeTarget::Gpu => "gpu",
                            settings::ComputeTarget::Cpu => "cpu",
                        };
                        if model_is_installed(&s.models_folder, model) {
                            if let Ok(worker) = worker::WorkerHandle::start(
                                &model.id,
                                default_variant,
                                s.models_folder.to_str().unwrap_or(""),
                                compute,
                            ) {
                                *state.worker.lock().await = Some(worker);
                                *state.last_activity.lock().await = Instant::now();
                                h.emit("engine", serde_json::json!({
                                    "model": model.id,
                                    "variant": default_variant,
                                    "state": "warm",
                                })).ok();
                            }
                        }
                    });
                }
            }

            // Spawn HW monitoring task
            let hw_handle = handle.clone();
            let hw_state = hw_info.clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

                loop {
                    interval.tick().await;

                    // Re-read GPU stats (on Windows via NVML)
                    let hw = hw::detect();
                    let payload = serde_json::json!({
                        "device": hw.devices.first().map(|d| d.name.clone()).unwrap_or_else(|| "CPU".into()),
                        "vram": if let Some(gpu) = hw.devices.iter().find(|d| d.vram_total_mb.is_some()) {
                            serde_json::json!({
                                "used": gpu.vram_used_mb,
                                "total": gpu.vram_total_mb,
                            })
                        } else {
                            serde_json::Value::Null
                        },
                        "ram": serde_json::json!({
                            "used": hw.ram_used,
                            "total": hw.ram_total,
                        }),
                    });

                    let _ = hw_handle.emit("hw", payload);
                }
            });

            // Spawn idle unload monitor
            let idle_handle = handle.clone();
            tauri::async_runtime::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

                loop {
                    interval.tick().await;

                    let state: tauri::State<AppState> = idle_handle.state();
                    let last = *state.last_activity.lock().await;
                    let idle_secs = last.elapsed().as_secs();

                    let s = state.settings.lock().await;
                    if let settings::IdleUnload::Minutes(mins) = s.idle_unload_minutes {
                        if idle_secs >= mins as u64 * 60 {
                            log::info!("Idle unload triggered after {}s", idle_secs);
                            if let Some(worker) = state.worker.lock().await.take() {
                                worker.stop().ok();
                            }
                            idle_handle.emit("engine", serde_json::json!({
                                "state": "cold",
                            })).ok();
                        }
                    }
                }
            });

            Ok(())
        })
        // ── Commands ──────────────────────────────────────────────
        .invoke_handler(tauri::generate_handler![
            bootstrap,
            save_settings,
            models,
            install_model,
            remove_model,
            preload,
            unload,
            generate,
            cancel,
            import_sources,
            paste_source,
            frame_to_source,
            delete_source,
            run_tool,
            save_frame,
            save_frame_as,
            delete_frame,
            reveal,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Latent");
}
