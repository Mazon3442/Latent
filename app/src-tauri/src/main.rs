// No console window behind the app in release builds.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod caelestia;
mod engine;
mod hw;
mod install;
mod library;
mod models;
mod remote;
mod store;
mod worker;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use engine::{GenRequest, Manager};
use library::{Library, Source};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use store::Settings;
use tauri::{AppHandle, Emitter, Manager as _, RunEvent, State};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_updater::UpdaterExt;

struct Dirs {
    resources: PathBuf,
    logs: PathBuf,
    settings: PathBuf,
}

struct App {
    dirs: Dirs,
    settings: Mutex<Settings>,
    library: Library,
    engines: Arc<Manager>,
    hw: Mutex<hw::Hardware>,
    remote: remote::Remote,
}

type Shared = Arc<App>;

impl App {
    fn paths(&self) -> engine::Paths {
        // Separate statements: each lock is released before the next one is taken.
        let home = PathBuf::from(&self.settings.lock().unwrap().models_dir);
        let device = self.device();
        let device_env = if device == "rocm" {
            self.hw.lock().unwrap().rocm_env.iter().filter(|(k, _)| std::env::var_os(k).is_none()).cloned().collect()
        } else {
            Vec::new()
        };
        engine::Paths { home, workers: self.dirs.resources.join("engines"), logs: self.dirs.logs.clone(), device, device_env }
    }

    fn device(&self) -> String {
        let setting = self.settings.lock().unwrap().device.clone();
        let hw = self.hw.lock().unwrap().clone();
        hw::resolve_device(&setting, &hw)
    }

    /// The variant of `m` to run: the one asked for, or the fastest that fits this PC.
    fn variant(&self, m: &models::Model, id: Option<&str>) -> &'static models::Variant {
        let hw = self.hw.lock().unwrap().clone();
        let lightest = self.settings.lock().unwrap().low_vram;
        m.variant(id, self.device() != "cpu", hw.vram_total, hw.ram_total, lightest)
    }

    /// The catalog with what this PC has: installed or not, whether each variant fits, and which one runs.
    fn models(&self) -> Value {
        let home = PathBuf::from(&self.settings.lock().unwrap().models_dir);
        let hw = self.hw.lock().unwrap().clone();
        let on_gpu = self.device() != "cpu";
        models::CATALOG
            .iter()
            .map(|m| {
                let mut v = serde_json::to_value(m).unwrap_or_default();
                v["installed"] = install::is_installed(&home, m).into();
                v["sizes"] = m.sizes.iter().map(|(a, w, h)| json!({ "aspect": a, "width": w, "height": h })).collect();
                for (j, variant) in m.variants.iter().enumerate() {
                    v["variants"][j]["fits"] = models::fits(variant, on_gpu, hw.vram_total, hw.ram_total).into();
                }
                v["recommended"] = self.variant(m, None).id.into();
                v
            })
            .collect()
    }
}

/// Long work runs off the async runtime so the interface never waits on it.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f).await.map_err(|e| e.to_string())?
}

// ---------------------------------------------------------------- commands

#[tauri::command]
fn bootstrap(handle: AppHandle, app: State<'_, Shared>) -> Value {
    // Copy everything out first: locks taken inside json!() would be held until the end of it.
    let settings = app.settings.lock().unwrap().clone();
    let hardware = app.hw.lock().unwrap().clone();
    json!({
        "version": handle.package_info().version.to_string(),
        "settings": settings,
        "hw": hardware,
        "device": app.device(),
        "models": app.models(),
        "sources": app.library.sources(),
        "frames": app.library.frames(),
        "resident": app.engines.resident(),
        "logs": app.dirs.logs,
        "platform": std::env::consts::OS,
        "caelestia": caelestia::read(),
    })
}

#[tauri::command]
fn save_settings(app: State<'_, Shared>, settings: Settings) -> Result<Settings, String> {
    store::save(&app.dirs.settings, &settings)?;
    *app.settings.lock().unwrap() = settings.clone();
    Ok(settings)
}

#[tauri::command]
async fn install_model(handle: AppHandle, app: State<'_, Shared>, id: String) -> Result<Value, String> {
    let app = app.inner().clone();
    blocking(move || {
        let m = models::find(&id)?;
        let (home, token) = {
            let s = app.settings.lock().unwrap();
            (PathBuf::from(&s.models_dir), s.hf_token.clone())
        };
        install::install(Some(&handle), &home, &app.dirs.logs, m, &app.device(), &token).inspect_err(|e| {
            // If uv is what's missing, hand the interface the command to copy for this platform.
            let command = (!install::uv_available()).then(install::uv_install_command);
            let _ = handle.emit("install", json!({ "id": id, "phase": "failed", "label": e, "command": command }));
        })?;
        Ok(app.models())
    })
    .await
}

#[tauri::command]
async fn remove_model(handle: AppHandle, app: State<'_, Shared>, id: String) -> Result<Value, String> {
    let app = app.inner().clone();
    blocking(move || {
        let m = models::find(&id)?;
        if app.engines.resident().as_deref() == Some(m.id) {
            app.engines.unload(&handle);
        }
        let home = PathBuf::from(&app.settings.lock().unwrap().models_dir);
        if !install::is_installed(&home, m) {
            return Err(format!("{} wasn't downloaded by Latent, so Latent won't delete its files.", m.name));
        }
        install::remove(&home, m)?;
        Ok(app.models())
    })
    .await
}

#[tauri::command]
async fn preload(handle: AppHandle, app: State<'_, Shared>, model: String, variant: Option<String>) -> Result<(), String> {
    let app = app.inner().clone();
    blocking(move || {
        let m = models::find(&model)?;
        app.engines.preload(&handle, &app.paths(), m, app.variant(m, variant.as_deref()).id)
    })
    .await
}

#[tauri::command]
async fn unload(handle: AppHandle, app: State<'_, Shared>) -> Result<(), String> {
    let app = app.inner().clone();
    blocking(move || {
        app.engines.unload(&handle);
        Ok(())
    })
    .await
}

/// Starts a batch and returns its id at once; frames arrive as events.
#[tauri::command]
fn generate(handle: AppHandle, app: State<'_, Shared>, mut request: GenRequest) -> Result<Value, String> {
    let m = models::find(&request.model)?;
    request.check(m)?;
    request.variant = Some(app.variant(m, request.variant.as_deref()).id.to_string());
    let sources = request.sources.iter().map(|id| app.library.source(id).map(|s| s.path)).collect::<Result<Vec<_>, _>>()?;
    if !app.engines.claim() {
        return Err("Still developing the last batch. Esc cancels it.".into());
    }
    let batch = format!("b{}", library::now_ms());
    {
        let mut s = app.settings.lock().unwrap();
        if s.last_model != m.id {
            s.last_model = m.id.to_string();
            let _ = store::save(&app.dirs.settings, &*s);
        }
    }
    let app = app.inner().clone();
    let id = batch.clone();
    std::thread::spawn(move || app.engines.run_batch(&handle, &app.paths(), &app.library, &id, &request, &sources));
    Ok(json!({ "batch": batch }))
}

#[tauri::command]
fn cancel(app: State<'_, Shared>, batch: String) {
    app.engines.cancel(&batch);
}

#[tauri::command]
async fn import_sources(handle: AppHandle, app: State<'_, Shared>, paths: Option<Vec<String>>) -> Result<Vec<Source>, String> {
    let app = app.inner().clone();
    blocking(move || {
        let paths: Vec<PathBuf> = match paths {
            Some(p) => p.into_iter().map(PathBuf::from).collect(),
            None => handle
                .dialog()
                .file()
                .set_title("Import photos")
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .blocking_pick_files()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|p| p.into_path().ok())
                .collect(),
        };
        paths.iter().map(|p| app.library.import(p)).collect()
    })
    .await
}

#[tauri::command]
async fn paste_source(app: State<'_, Shared>, png: String) -> Result<Source, String> {
    let app = app.inner().clone();
    blocking(move || {
        let bytes = STANDARD.decode(png.split(',').next_back().unwrap_or_default()).map_err(|e| e.to_string())?;
        app.library.paste(&bytes)
    })
    .await
}

#[tauri::command]
async fn frame_to_source(app: State<'_, Shared>, frame: String) -> Result<Source, String> {
    let app = app.inner().clone();
    blocking(move || app.library.source_from_frame(&frame)).await
}

#[tauri::command]
fn delete_source(app: State<'_, Shared>, id: String) -> Result<(), String> {
    app.library.delete_source(&id)
}

fn unique(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let mut p = dir.join(format!("{stem}.{ext}"));
    let mut n = 2;
    while p.exists() {
        p = dir.join(format!("{stem} ({n}).{ext}"));
        n += 1;
    }
    p
}

/// PNG is copied as it is, so the recipe and the AI-generated mark in it are kept. JPEG is re-encoded: smaller, but it
/// has no place for them.
fn export(src: &str, dst: &Path) -> Result<(), String> {
    let ext = dst.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).unwrap_or_default();
    let fail = |e: String| format!("can't save to {}: {e}", dst.display());
    match ext.as_str() {
        "jpg" | "jpeg" => {
            let image = image::open(src).map_err(|e| e.to_string())?.to_rgb8();
            let file = std::fs::File::create(dst).map_err(|e| fail(e.to_string()))?;
            image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::BufWriter::new(file), 92)
                .encode_image(&image)
                .map_err(|e| fail(e.to_string()))
        }
        "png" | "" => std::fs::copy(src, dst).map(|_| ()).map_err(|e| fail(e.to_string())),
        // Only PNG carries the recipe and the AI-generated mark, so no other format is written under its name.
        other => Err(fail(format!("saving as .{other} isn't supported; use PNG or JPEG"))),
    }
}

fn save_ext(app: &App) -> &'static str {
    if app.settings.lock().unwrap().format == "JPEG" { "jpg" } else { "png" }
}

/// A file name from the first words of the prompt and the seed: "lighthouse-at-dusk-41.png".
fn file_stem(prompt: &str, seed: i64) -> String {
    let words: Vec<String> = prompt
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(5)
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() { format!("latent-{seed}") } else { format!("{}-{seed}", words.join("-")) }
}

#[tauri::command]
async fn save_frame(app: State<'_, Shared>, id: String) -> Result<String, String> {
    let app = app.inner().clone();
    blocking(move || {
        let f = app.library.frame(&id)?;
        let dir = PathBuf::from(&app.settings.lock().unwrap().save_dir);
        std::fs::create_dir_all(&dir).map_err(|e| format!("can't create {}: {e}", dir.display()))?;
        let dst = unique(&dir, &file_stem(f.recipe["prompt"].as_str().unwrap_or_default(), f.seed), save_ext(&app));
        export(&f.path, &dst)?;
        let dst = dst.to_string_lossy().to_string();
        app.library.mark_saved(&id, &dst);
        Ok(dst)
    })
    .await
}

#[tauri::command]
async fn save_frame_as(handle: AppHandle, app: State<'_, Shared>, id: String) -> Result<Option<String>, String> {
    let app = app.inner().clone();
    blocking(move || {
        let f = app.library.frame(&id)?;
        let ext = save_ext(&app);
        let mut dialog = handle
            .dialog()
            .file()
            .set_title("Save frame as")
            .set_file_name(format!("{}.{ext}", file_stem(f.recipe["prompt"].as_str().unwrap_or_default(), f.seed)));
        // The format from Settings comes first; the other is one click away.
        for (name, e) in if ext == "jpg" { [("JPEG", "jpg"), ("PNG", "png")] } else { [("PNG", "png"), ("JPEG", "jpg")] } {
            dialog = dialog.add_filter(name, &[e]);
        }
        let Some(picked) = dialog.blocking_save_file() else { return Ok(None) };
        let dst = picked.into_path().map_err(|e| e.to_string())?;
        export(&f.path, &dst)?;
        let dst = dst.to_string_lossy().to_string();
        app.library.mark_saved(&id, &dst);
        Ok(Some(dst))
    })
    .await
}

// ---------------------------------------------------------------- remote access

#[tauri::command]
fn remote_status(app: State<'_, Shared>) -> Value {
    app.remote.status(&app)
}

#[tauri::command]
async fn remote_start(handle: AppHandle, mode: String) -> Result<Value, String> {
    blocking(move || remote::start(&handle, &mode)).await
}

#[tauri::command]
async fn remote_stop(handle: AppHandle) -> Result<Value, String> {
    blocking(move || Ok(remote::stop(&handle))).await
}

#[tauri::command]
fn remote_new_link(handle: AppHandle) -> Result<Value, String> {
    remote::new_link(&handle)
}

/// Opens one of Tailscale's pages (download, turning on HTTPS) in the browser, for the sharing setup guide.
/// Only Tailscale's sites, so the window can't be made to open anything else.
#[tauri::command]
fn open_tailscale_page(url: String) -> Result<(), String> {
    let allowed = ["https://tailscale.com/", "https://login.tailscale.com/"].iter().any(|p| url.starts_with(p));
    if !allowed || url.chars().any(|c| c.is_whitespace() || c == '"') {
        return Err("Latent only opens Tailscale's own pages.".into());
    }
    open_in_browser(&url)
}

/// Runs a UI command for a browser on another device. Commands that open windows on this PC, take its file paths or
/// use its secrets aren't offered; remote.rs has browser versions of the ones that matter (upload, download).
fn remote_call(h: &AppHandle, cmd: &str, args: Value) -> Result<Value, String> {
    use tauri::async_runtime::block_on;
    fn arg<T: serde::de::DeserializeOwned>(args: &Value, k: &str) -> Result<T, String> {
        serde_json::from_value(args.get(k).cloned().unwrap_or(Value::Null)).map_err(|e| format!("{k}: {e}"))
    }
    fn to<T: serde::Serialize>(r: Result<T, String>) -> Result<Value, String> {
        r.and_then(|v| serde_json::to_value(v).map_err(|e| e.to_string()))
    }
    let app = || h.state::<Shared>();
    match cmd {
        "bootstrap" => {
            let mut b = bootstrap(h.clone(), app());
            b["remote"] = true.into();
            b["settings"]["hfToken"] = "".into(); // the host's secret stays on the host
            b["logs"] = "".into();
            Ok(b)
        }
        "save_settings" => {
            // Folders and the token belong to the host PC, so a browser can't change them.
            let mut s: Settings = arg(&args, "settings")?;
            let host = app().settings.lock().unwrap().clone();
            (s.models_dir, s.save_dir, s.hf_token) = (host.models_dir, host.save_dir, host.hf_token);
            to(save_settings(app(), s))
        }
        "install_model" => to(block_on(install_model(h.clone(), app(), arg(&args, "id")?))),
        "remove_model" => to(block_on(remove_model(h.clone(), app(), arg(&args, "id")?))),
        "preload" => to(block_on(preload(h.clone(), app(), arg(&args, "model")?, arg(&args, "variant")?)).map(|_| Value::Null)),
        "unload" => to(block_on(unload(h.clone(), app())).map(|_| Value::Null)),
        "generate" => to(generate(h.clone(), app(), arg(&args, "request")?)),
        "cancel" => {
            cancel(app(), arg(&args, "batch")?);
            Ok(Value::Null)
        }
        "paste_source" => to(block_on(paste_source(app(), arg(&args, "png")?))),
        "frame_to_source" => to(block_on(frame_to_source(app(), arg(&args, "frame")?))),
        "delete_source" => to(delete_source(app(), arg(&args, "id")?).map(|_| Value::Null)),
        "delete_frame" => {
            delete_frame(app(), arg(&args, "id")?);
            Ok(Value::Null)
        }
        "models_disk" => to(block_on(models_disk(app()))),
        "ui_error" => {
            ui_error(app(), format!("[remote] {}", arg::<String>(&args, "message")?));
            Ok(Value::Null)
        }
        other => Err(format!("{other} only works on the PC running Latent.")),
    }
}

#[tauri::command]
fn delete_frame(app: State<'_, Shared>, id: String) {
    app.library.delete_frame(&id);
}

#[tauri::command]
async fn pick_folder(handle: AppHandle, title: String) -> Result<Option<String>, String> {
    blocking(move || {
        Ok(handle.dialog().file().set_title(title).blocking_pick_folder().and_then(|p| p.into_path().ok()).map(|p| p.to_string_lossy().to_string()))
    })
    .await
}

/// Show a saved file in the file manager.
#[tauri::command]
fn reveal(path: String) {
    #[cfg(windows)]
    let _ = std::process::Command::new("explorer").arg(format!("/select,{path}")).spawn();
    #[cfg(unix)]
    {
        let dir = Path::new(&path).parent().map(Path::to_path_buf).unwrap_or_default();
        let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
    }
}

/// Open a help link in the browser. Only Hugging Face and GitHub pages: the interface can't send the user anywhere else.
#[tauri::command]
fn open_url(url: String) -> Result<(), String> {
    if !allowed_url(&url) {
        return Err("Latent only opens Hugging Face and GitHub pages.".into());
    }
    open_in_browser(&url)
}

/// The caller has already checked the address.
fn open_in_browser(url: &str) -> Result<(), String> {
    #[cfg(windows)]
    let opened = std::process::Command::new("explorer").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let opened = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let opened = std::process::Command::new("xdg-open").arg(url).spawn();
    opened.map(|_| ()).map_err(|e| format!("couldn't open the browser: {e}"))
}

fn allowed_url(url: &str) -> bool {
    let allowed = ["https://huggingface.co/", "https://github.com/", "https://docs.astral.sh/"];
    allowed.iter().any(|p| url.starts_with(p)) && !url.contains(|c: char| c.is_whitespace() || c == '"' || c.is_control())
}

/// Disk used by the models folder: the Python environment and every downloaded model.
#[tauri::command]
async fn models_disk(app: State<'_, Shared>) -> Result<u64, String> {
    let home = PathBuf::from(&app.settings.lock().unwrap().models_dir);
    blocking(move || Ok(install::folder_bytes(&home))).await
}

/// Ask Hugging Face who a token belongs to, and whether it can download each model that needs an account. Runs in
/// Latent's Python environment; the token goes in the environment, not on a command line other programs can read.
#[tauri::command]
async fn test_hf_token(app: State<'_, Shared>, token: String) -> Result<Value, String> {
    let home = PathBuf::from(&app.settings.lock().unwrap().models_dir);
    blocking(move || {
        let py = install::python(&home);
        if !py.is_file() {
            return Err("The check runs in Latent's Python environment, which arrives with the first model. Download any model first.".into());
        }
        let repos: Vec<(&str, &str)> = models::CATALOG.iter().filter(|m| m.account).map(|m| (m.id, m.repo)).collect();
        let script = r#"
import json, os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
try:
    name = api.whoami()["name"]
except Exception as e:
    print(json.dumps({"error": "Hugging Face didn't accept this token. Check it was copied whole, and that it hasn't been deleted."})); sys.exit()
access = {}
for model_id, repo in json.loads(sys.argv[1]):
    try:
        api.auth_check(repo); access[model_id] = True
    except Exception:
        access[model_id] = False
print(json.dumps({"name": name, "access": access}))
"#;
        let mut cmd = std::process::Command::new(&py);
        cmd.arg("-c").arg(script).arg(serde_json::to_string(&repos).unwrap_or_default()).env("HF_TOKEN", token.trim());
        worker::hidden(&mut cmd);
        let out = cmd.output().map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&out.stdout);
        let v: Value = text.lines().rev().find_map(|l| serde_json::from_str(l).ok())
            .ok_or_else(|| "Couldn't reach Hugging Face. Check the internet connection and try again.".to_string())?;
        if let Some(e) = v["error"].as_str() {
            return Err(e.to_string());
        }
        Ok(v)
    })
    .await
}

#[tauri::command]
fn ui_error(app: State<'_, Shared>, message: String) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(app.dirs.logs.join("ui.log")) {
        let _ = writeln!(f, "{} {message}", library::now_ms());
    }
}

// ---------------------------------------------------------------- updates

/// The package manager that owns this install when a distro package built it (set by packaging/arch/PKGBUILD).
/// Such installs hear about new versions but get them from that package manager, not from the updater.
const PACKAGE_MANAGER: Option<&str> = option_env!("LATENT_PACKAGE");

/// The newest GitHub release, if it's newer than this build. Development builds never offer one.
#[tauri::command]
async fn check_update(handle: AppHandle) -> Result<Option<Value>, String> {
    if cfg!(debug_assertions) {
        return Ok(None);
    }
    let update = handle.updater().map_err(|e| e.to_string())?.check().await.map_err(|e| format!("Couldn't check for updates: {e}"))?;
    Ok(update.map(|u| json!({
        "version": u.version, "current": u.current_version, "notes": u.body.clone().unwrap_or_default(),
        "packageManager": PACKAGE_MANAGER,
    })))
}

/// Download the newest release, install it and restart into it. Progress arrives as "update-progress" events.
#[tauri::command]
async fn install_update(handle: AppHandle) -> Result<(), String> {
    if let Some(manager) = PACKAGE_MANAGER {
        return Err(format!("This copy of Latent was installed with {manager}. Update it there."));
    }
    let update = handle
        .updater()
        .map_err(|e| e.to_string())?
        .check()
        .await
        .map_err(|e| format!("Couldn't check for updates: {e}"))?
        .ok_or("Latent is already up to date.")?;
    let h = handle.clone();
    let mut done = 0u64;
    update
        .download_and_install(
            move |chunk, total| {
                done += chunk as u64;
                let _ = h.emit("update-progress", json!({ "done": done, "total": total }));
            },
            || {},
        )
        .await
        .map_err(|e| format!("Couldn't install the update: {e}"))?;
    handle.restart()
}

// ---------------------------------------------------------------- setup

fn resources_dir(handle: &AppHandle) -> PathBuf {
    let bundled = handle.path().resource_dir().unwrap_or_default();
    if bundled.join("engines").join("diffusion_worker.py").is_file() {
        return bundled;
    }
    // Running from the repo (cargo run / tauri dev): use the source folders.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// `latent --install <model>` downloads a model without opening the window (used for testing and scripting).
fn cli_install(id: &str) -> ! {
    // Same folders Tauri uses: %APPDATA% / %LOCALAPPDATA% on Windows, $XDG_DATA_HOME on Linux, and
    // ~/Library/Application Support on macOS (where app_data_dir and app_local_data_dir both point).
    let (data, local) = if cfg!(windows) {
        (std::env::var("APPDATA").unwrap_or_default(), std::env::var("LOCALAPPDATA").unwrap_or_default())
    } else if cfg!(target_os = "macos") {
        let base = format!("{}/Library/Application Support", std::env::var("HOME").unwrap_or_default());
        (base.clone(), base)
    } else {
        let d = std::env::var("XDG_DATA_HOME").unwrap_or_else(|_| format!("{}/.local/share", std::env::var("HOME").unwrap_or_default()));
        (d.clone(), d)
    };
    let data = PathBuf::from(data).join("com.mazon.latent");
    let logs = PathBuf::from(local).join("com.mazon.latent").join("logs");
    let _ = std::fs::create_dir_all(&logs);
    let settings: Settings = store::load(&data.join("settings.json"));
    let device = hw::resolve_device(&settings.device, &hw::snapshot(&mut sysinfo::System::new()));
    let result = models::find(id).and_then(|m| {
        println!("installing {} for {device} into {}", m.name, settings.models_dir);
        install::install(None, Path::new(&settings.models_dir), &logs, m, &device, &settings.hf_token)
    });
    match result {
        Ok(()) => {
            println!("installed {id}");
            std::process::exit(0)
        }
        Err(e) => {
            eprintln!("install failed: {e}");
            std::process::exit(1)
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--install") {
        cli_install(args.get(i + 1).map(String::as_str).unwrap_or(""));
    }
    // WebKitGTK's DMA-BUF renderer shows a blank window with NVIDIA's Linux driver on Wayland.
    #[cfg(target_os = "linux")]
    if Path::new("/proc/driver/nvidia").exists() && std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|tauri_app| {
            let handle = tauri_app.handle().clone();
            // LATENT_DATA_DIR runs a separate copy with its own settings, sources, frames and sharing secret (for testing:
            // a second copy would otherwise clear the first one's unsaved frames).
            let (data, local) = match std::env::var_os("LATENT_DATA_DIR") {
                Some(dir) => (PathBuf::from(&dir), PathBuf::from(dir).join("local")),
                None => (handle.path().app_data_dir()?, handle.path().app_local_data_dir()?),
            };
            let dirs = Dirs { resources: resources_dir(&handle), logs: local.join("logs"), settings: data.join("settings.json") };
            std::fs::create_dir_all(&dirs.logs)?;
            let settings: Settings = store::load(&dirs.settings);
            let library = Library::open(&data, local.join("frames"))?;
            // Read the hardware once before the window asks, so models aren't judged against an empty GPU.
            let mut sys = sysinfo::System::new();
            let first = hw::snapshot(&mut sys);
            let app: Shared = Arc::new(App {
                dirs,
                settings: Mutex::new(settings),
                library,
                engines: Arc::new(Manager::default()),
                hw: Mutex::new(first),
                remote: remote::Remote::default(),
            });
            remote::Remote::init(&handle);
            tauri_app.manage(app.clone());

            // Hyprland tiles windows and draws its own borders, so a GTK title bar would just waste space.
            #[cfg(target_os = "linux")]
            if let Some(win) = handle.get_webview_window("main") {
                let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_default().to_lowercase();
                if desktop.contains("hyprland") || std::env::var_os("HYPRLAND_INSTANCE_SIGNATURE").is_some() {
                    let _ = win.set_decorations(false);
                }
            }

            // Hardware every 2 s, and the idle-unload check.
            let h = handle.clone();
            let mut scheme_seen = caelestia::modified();
            std::thread::spawn(move || loop {
                // Caelestia rewrites its scheme when the wallpaper changes; Latent re-colors with it. If the file
                // is caught mid-write (invalid JSON), leave scheme_seen alone so the next 2 s tick reads it again.
                let scheme_now = caelestia::modified();
                if scheme_now != scheme_seen {
                    match caelestia::read() {
                        Some(scheme) => {
                            scheme_seen = scheme_now;
                            let _ = h.emit("caelestia", scheme);
                        }
                        // Deleted: go back to Latent's own theme instead of keeping the last colors it saw.
                        None if !caelestia::present() => {
                            scheme_seen = None;
                            let _ = h.emit("caelestia", serde_json::Value::Null);
                        }
                        None => {}
                    }
                }
                let snap = hw::snapshot(&mut sys);
                *app.hw.lock().unwrap() = snap.clone();
                let minutes = app.settings.lock().unwrap().idle_minutes();
                app.engines.idle_check(&h, minutes);
                let _ = h.emit("hw", json!({
                    "hw": snap,
                    "resident": app.engines.resident(),
                    "idleLeft": app.engines.idle_seconds_left(minutes),
                }));
                std::thread::sleep(Duration::from_secs(2));
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            bootstrap, save_settings, install_model, remove_model, preload, unload, generate, cancel,
            import_sources, paste_source, frame_to_source, delete_source, save_frame, save_frame_as, delete_frame,
            pick_folder, reveal, ui_error, check_update, install_update, open_url, models_disk, test_hf_token,
            remote_status, remote_start, remote_stop, remote_new_link, open_tailscale_page
        ])
        .build(tauri::generate_context!())
        .expect("error while building Latent")
        .run(|handle, event| {
            if let RunEvent::Exit = event {
                if let Some(app) = handle.try_state::<Shared>() {
                    remote::stop(handle); // also takes Latent's route out of Tailscale
                    app.engines.unload(handle);
                    app.library.clear_frames();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_help_pages_open() {
        assert!(allowed_url("https://huggingface.co/settings/tokens"));
        assert!(allowed_url("https://github.com/Mazon3442/Latent/releases"));
        assert!(!allowed_url("https://huggingface.co.evil.example/"), "a look-alike host");
        assert!(!allowed_url("http://huggingface.co/"), "plain http");
        assert!(!allowed_url("file:///C:/Windows/System32/calc.exe"));
        assert!(!allowed_url("https://github.com/x \"--some-flag\""), "no extra arguments");
    }

    #[test]
    fn jpeg_export_reencodes_and_png_copies() {
        let dir = std::env::temp_dir().join(format!("latent-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("frame.png");
        image::RgbImage::from_pixel(32, 16, image::Rgb([200, 120, 40])).save(&src).unwrap();
        let src = src.to_string_lossy().to_string();
        export(&src, &dir.join("a.jpg")).unwrap();
        let jpg = image::open(dir.join("a.jpg")).unwrap();
        assert_eq!((jpg.width(), jpg.height()), (32, 16));
        assert_eq!(std::fs::read(dir.join("a.jpg")).unwrap()[..2], [0xFF, 0xD8], "a real JPEG");
        export(&src, &dir.join("b.png")).unwrap();
        assert_eq!(std::fs::read(dir.join("b.png")).unwrap(), std::fs::read(&src).unwrap(), "PNG is copied byte for byte");
        assert!(export(&src, &dir.join("c.webp")).is_err(), "no format is written under a name that isn't its own");
        assert!(!dir.join("c.webp").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_names_come_from_the_prompt() {
        assert_eq!(file_stem("A lighthouse at dusk, wet stone, film grain", 41), "a-lighthouse-at-dusk-wet-41");
        assert_eq!(file_stem("  ", 7), "latent-7");
        assert_eq!(file_stem("../../etc/passwd", 3), "etc-passwd-3", "no path characters survive");
    }
}
