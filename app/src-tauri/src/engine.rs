//! The model manager. Only one model is in memory at a time: loading one unloads the other first, because two
//! diffusion models don't fit in 16 GB together. Batches run one frame at a time on a background thread, and their
//! progress and results reach the interface as events.

use crate::install;
use crate::library::{now_ms, Frame, Library};
use crate::models::{self, Model};
use crate::worker::{self, Worker};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::process::{Child, ChildStdin};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter};

const LOAD_TIMEOUT: Duration = Duration::from_secs(10 * 60); // a cold disk read of 30 GB of weights
/// Longest silence between two events of a running frame (the first step includes encoding the prompt).
const STEP_TIMEOUT: Duration = Duration::from_secs(5 * 60);
/// How long Cancel waits for the worker to stop at a step boundary before killing it.
const CANCEL_GRACE: Duration = Duration::from_secs(10);

pub struct Paths {
    /// Models folder: the Python environment and the weights (Settings -> Models folder).
    pub home: PathBuf,
    /// Worker scripts shipped with the app.
    pub workers: PathBuf,
    pub logs: PathBuf,
    /// Where models run: "cuda" | "rocm" | "xpu" | "cpu".
    pub device: String,
    pub device_env: Vec<(String, String)>,
}

#[derive(Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct GenRequest {
    pub model: String,
    #[serde(default)]
    pub variant: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub negative: String,
    /// Source ids; empty makes a new image.
    #[serde(default)]
    pub sources: Vec<String>,
    pub width: u32,
    pub height: u32,
    pub count: u32,
    /// None picks a random one; frame i uses seed + i.
    pub seed: Option<i64>,
    #[serde(default)]
    pub options: Map<String, Value>,
}

impl GenRequest {
    /// Everything that arrives from the interface is checked here, before it can reach the GPU.
    pub fn check(&self, m: &Model) -> Result<(), String> {
        if self.prompt.trim().is_empty() {
            return Err("Write a prompt first".into());
        }
        if !(1..=8).contains(&self.count) {
            return Err("A batch is 1 to 8 frames".into());
        }
        let largest = m.sizes.iter().map(|s| s.1.max(s.2)).max().unwrap_or(2048);
        for side in [self.width, self.height] {
            if !(256..=largest).contains(&side) || side % m.grid != 0 {
                return Err(format!("{} needs sizes from 256 to {largest} px in steps of {}", m.name, m.grid));
            }
        }
        if !self.sources.is_empty() && !m.can.edit {
            return Err(format!("{} makes new images; pick a model that can edit", m.name));
        }
        if let Some(steps) = self.options.get("steps") {
            let s = steps.as_f64().ok_or("steps must be a number")?;
            if s < m.options.steps.min || s > m.options.steps.max {
                return Err(format!("{} runs {} to {} steps", m.name, m.options.steps.min, m.options.steps.max));
            }
        }
        Ok(())
    }
}

#[derive(Default)]
pub struct Manager {
    slot: Mutex<Option<(String, Worker)>>,
    sender: Mutex<Option<Arc<Mutex<ChildStdin>>>>,
    killer: Mutex<Option<Arc<Mutex<Child>>>>,
    last_used: Mutex<Option<Instant>>,
    busy: AtomicBool,
    /// The batch running now, and its frame request in flight (None while the model loads or between frames),
    /// so Cancel knows what to stop.
    running: Mutex<Option<(String, Option<String>)>>,
    cancelled: AtomicBool,
}

fn emit_engine(app: &AppHandle, model: &str, state: &str, message: Option<&str>) {
    let _ = app.emit("engine", json!({ "model": model, "state": state, "message": message }));
}

impl Manager {
    pub fn resident(&self) -> Option<String> {
        self.slot.try_lock().ok().and_then(|s| s.as_ref().map(|(id, _)| id.clone()))
    }

    fn load(&self, paths: &Paths, m: &Model, variant: &str) -> Result<Worker, String> {
        if !install::is_installed(&paths.home, m) {
            return Err(format!("{} isn't downloaded yet (Models)", m.name));
        }
        let mut envs = vec![
            ("LATENT_DEVICE", crate::hw::torch_device(&paths.device).to_string()),
            ("HF_HUB_OFFLINE", "1".to_string()), // everything is on disk; loading never touches the network
            ("HF_HUB_DISABLE_SYMLINKS_WARNING", "1".to_string()),
        ];
        if paths.device == "mps" {
            // Metal doesn't implement every op diffusers uses; let those fall back to the CPU instead of failing.
            envs.push(("PYTORCH_ENABLE_MPS_FALLBACK", "1".to_string()));
        }
        envs.extend(paths.device_env.iter().map(|(k, v)| (k.as_str(), v.clone())));
        let dir = install::model_dir(&paths.home, m).to_string_lossy().to_string();
        let v = m.variants.iter().find(|v| v.id == variant).unwrap_or(&m.variants[0]);
        Worker::start(
            &install::python(&paths.home),
            &paths.workers.join("diffusion_worker.py"),
            &[m.family, &dir, v.offload, v.weights],
            &envs,
            &paths.logs.join(format!("{}.log", m.id)),
            LOAD_TIMEOUT,
        )
    }

    /// Make `m` the resident model, unloading whatever was there.
    fn ensure(&self, app: &AppHandle, paths: &Paths, slot: &mut Option<(String, Worker)>, m: &Model, variant: &str) -> Result<(), String> {
        if let Some((id, _)) = slot.as_ref() {
            if id == m.id {
                return Ok(());
            }
            let old = id.clone();
            emit_engine(app, &old, "unloading", None);
            *slot = None; // Drop kills the worker and frees its memory
            *self.sender.lock().unwrap() = None;
            *self.killer.lock().unwrap() = None;
            emit_engine(app, &old, "cold", None);
        }
        emit_engine(app, m.id, "loading", None);
        match self.load(paths, m, variant) {
            Ok(w) => {
                *self.sender.lock().unwrap() = Some(w.sender());
                *self.killer.lock().unwrap() = Some(w.killer());
                if w.hello["fallback"].is_object() {
                    let _ = app.emit("fallback", json!({ "model": m.id, "fallback": w.hello["fallback"] }));
                }
                *slot = Some((m.id.to_string(), w));
                *self.last_used.lock().unwrap() = Some(Instant::now());
                emit_engine(app, m.id, "warm", None);
                Ok(())
            }
            Err(e) => {
                emit_engine(app, m.id, "cold", Some(&e));
                Err(format!("Couldn't load {}: {e}", m.name))
            }
        }
    }

    pub fn preload(&self, app: &AppHandle, paths: &Paths, m: &Model, variant: &str) -> Result<(), String> {
        let mut slot = self.slot.lock().unwrap();
        self.ensure(app, paths, &mut slot, m, variant)
    }

    /// Claims the manager for one batch. False while another batch runs.
    pub fn claim(&self) -> bool {
        self.busy.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok()
    }

    /// Runs a claimed batch to the end. Every outcome is reported as events; nothing is returned.
    pub fn run_batch(&self, app: &AppHandle, paths: &Paths, lib: &Library, batch: &str, req: &GenRequest, sources: &[String]) {
        self.cancelled.store(false, Ordering::SeqCst);
        *self.running.lock().unwrap() = Some((batch.to_string(), None));
        let finish = |state: &str, error: Option<Value>| {
            let _ = app.emit("batch", json!({ "batch": batch, "state": state, "error": error }));
        };
        let m = match models::find(&req.model) {
            Ok(m) => m,
            Err(e) => return self.release(|| finish("failed", Some(json!({ "kind": "input", "message": e })))),
        };
        let variant = m.variants.iter().find(|v| Some(v.id) == req.variant.as_deref()).unwrap_or(&m.variants[0]);
        let mut slot = self.slot.lock().unwrap();
        if let Err(e) = self.ensure(app, paths, &mut slot, m, variant.id) {
            drop(slot);
            return self.release(|| finish("failed", Some(json!({ "kind": "model", "message": e }))));
        }
        let base_seed = req.seed.unwrap_or_else(|| (now_ms() % 2_147_483_647) as i64);
        let mut outcome = ("done", None);
        for i in 0..req.count {
            if self.cancelled.load(Ordering::SeqCst) {
                outcome = ("cancelled", None);
                break;
            }
            let id = format!("{batch}-{i}");
            let seed = base_seed + i as i64;
            let output = lib.frames_dir.join(format!("{id}.png")).to_string_lossy().to_string();
            let mut recipe = serde_json::to_value(req).unwrap_or_default();
            recipe["seed"] = seed.into();
            recipe["count"] = 1.into();
            let request = json!({
                "op": "generate", "id": id, "prompt": req.prompt, "negative": req.negative, "sources": sources,
                "width": req.width, "height": req.height, "seed": seed, "options": req.options,
                "output": output, "recipe": recipe,
            });
            *self.running.lock().unwrap() = Some((batch.to_string(), Some(id.clone())));
            emit_engine(app, m.id, "busy", None);
            let w = &slot.as_ref().unwrap().1;
            let result = w.stream(&request, STEP_TIMEOUT, |p| {
                let _ = app.emit("progress", json!({
                    "batch": batch, "index": i, "step": p["step"], "steps": p["steps"],
                    "preview": p["preview"], "secPerStep": p["sec_per_step"],
                }));
            });
            *self.running.lock().unwrap() = Some((batch.to_string(), None));
            match result {
                Ok(v) if v["event"] == "done" => {
                    let frame = Frame {
                        id: id.clone(),
                        batch: batch.to_string(),
                        index: i,
                        path: output,
                        thumb: v["thumb"].as_str().unwrap_or_default().to_string(),
                        width: req.width,
                        height: req.height,
                        seed,
                        recipe,
                        model: m.id.to_string(),
                        variant: variant.id.to_string(),
                        elapsed: v["elapsed"].as_f64().unwrap_or(0.0),
                        saved: None,
                        created_at: now_ms(),
                    };
                    lib.add_frame(frame.clone());
                    let _ = app.emit("frame", &frame);
                }
                Ok(v) if v["event"] == "cancelled" => {
                    outcome = ("cancelled", None);
                    break;
                }
                Ok(v) => {
                    let kind = v["kind"].as_str().unwrap_or("model");
                    let fix = (kind == "oom").then_some("Try a smaller size, or fewer frames per batch.");
                    outcome = ("failed", Some(json!({ "kind": kind, "message": v["message"], "fix": fix })));
                    break;
                }
                Err(e) => {
                    // The process died or hung (or Cancel had to kill it): it can't be trusted with the next frame.
                    *slot = None;
                    *self.sender.lock().unwrap() = None;
                    *self.killer.lock().unwrap() = None;
                    emit_engine(app, m.id, "cold", None);
                    outcome = if self.cancelled.load(Ordering::SeqCst) {
                        ("cancelled", None)
                    } else {
                        ("failed", Some(json!({ "kind": "model", "message": format!("{} {e}. Details are in {}.log.", m.name, m.id) })))
                    };
                    break;
                }
            }
        }
        if slot.is_some() {
            emit_engine(app, m.id, "warm", None);
        }
        drop(slot);
        self.release(|| finish(outcome.0, outcome.1));
    }

    fn release(&self, report: impl FnOnce()) {
        *self.running.lock().unwrap() = None;
        *self.last_used.lock().unwrap() = Some(Instant::now());
        self.busy.store(false, Ordering::SeqCst);
        report();
    }

    /// Stops a batch: the frame in flight stops at its next step and the model stays loaded. If the worker doesn't
    /// answer within the grace period, it's killed (and reloads on the next batch). While the model is still loading,
    /// the batch ends as soon as the load does.
    pub fn cancel(self: &Arc<Self>, batch: &str) {
        let Some((running, request)) = self.running.lock().unwrap().clone() else { return };
        if running != batch {
            return;
        }
        self.cancelled.store(true, Ordering::SeqCst);
        let Some(request) = request else { return };
        if let Some(stdin) = self.sender.lock().unwrap().as_ref() {
            let _ = worker::send(stdin, &json!({ "op": "cancel", "id": request }));
        }
        let me = self.clone();
        std::thread::spawn(move || {
            std::thread::sleep(CANCEL_GRACE);
            if me.running.lock().unwrap().as_ref().and_then(|(_, r)| r.as_ref()) == Some(&request) {
                if let Some(k) = me.killer.lock().unwrap().as_ref() {
                    if let Ok(mut c) = k.lock() {
                        worker::kill_tree(&mut c);
                    }
                }
            }
        });
    }

    pub fn unload(&self, app: &AppHandle) {
        if let Some(k) = self.killer.lock().unwrap().as_ref() {
            if self.busy.load(Ordering::SeqCst) {
                self.cancelled.store(true, Ordering::SeqCst);
                if let Ok(mut c) = k.lock() {
                    worker::kill_tree(&mut c);
                }
            }
        }
        let mut slot = self.slot.lock().unwrap();
        if let Some((id, _)) = slot.take() {
            emit_engine(app, &id, "unloading", None);
            *self.sender.lock().unwrap() = None;
            *self.killer.lock().unwrap() = None;
            emit_engine(app, &id, "cold", None);
        }
    }

    /// Called every few seconds: unloads the model after the idle time from Settings.
    pub fn idle_check(&self, app: &AppHandle, minutes: Option<u64>) {
        let Some(m) = minutes else { return };
        if self.busy.load(Ordering::SeqCst) {
            return;
        }
        let idle = self.last_used.lock().unwrap().map(|t| t.elapsed() > Duration::from_secs(m * 60)).unwrap_or(false);
        if !idle {
            return;
        }
        if let Ok(mut slot) = self.slot.try_lock() {
            if let Some((id, _)) = slot.take() {
                *self.sender.lock().unwrap() = None;
                *self.killer.lock().unwrap() = None;
                emit_engine(app, &id, "cold", None);
                let _ = app.emit("log", format!("Unloaded {id} after {m} idle minutes. Memory is free."));
            }
        }
    }

    pub fn idle_seconds_left(&self, minutes: Option<u64>) -> Option<u64> {
        let m = minutes?;
        let t = (*self.last_used.lock().unwrap())?;
        self.resident()?;
        Some((m * 60).saturating_sub(t.elapsed().as_secs()))
    }
}
