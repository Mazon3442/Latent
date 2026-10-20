/// Worker management: launch and communicate with the Python diffusion worker.
/// The worker is a subprocess that reads NDJSON from stdin and writes NDJSON to stdout.
/// Protocol: see docs/PLAN.md section 5 (worker protocol).

use serde::{Deserialize, Serialize};
use std::io::{BufRead as _, Write as _};
use std::process::{Child, ChildStdin, Command};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use tokio::sync::mpsc;
use tokio::task;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineState {
    Cold,
    Loading { elapsed: f64 },
    Warm,
    Busy,
    Unloading,
    Crashed { message: String },
}

impl std::fmt::Display for EngineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineState::Cold => write!(f, "cold"),
            EngineState::Loading { .. } => write!(f, "loading"),
            EngineState::Warm => write!(f, "warm"),
            EngineState::Busy => write!(f, "busy"),
            EngineState::Unloading => write!(f, "unloading"),
            EngineState::Crashed { message } => write!(f, "crashed: {}", message),
        }
    }
}

/// Message from worker to core.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event")]
pub enum WorkerEvent {
    #[serde(rename = "ready")]
    Ready { model: String, variant: String, device: String, fallback: Option<String>, vram: f64 },

    #[serde(rename = "progress")]
    Progress { id: String, step: usize, steps: usize, preview: Option<String>, sec_per_step: Option<f64> },

    #[serde(rename = "done")]
    Done { id: String, output: String, seed: i64, elapsed: f64 },

    #[serde(rename = "cancelled")]
    Cancelled { id: String },

    #[serde(rename = "error")]
    Error { id: String, kind: ErrorKind, message: String },

    /// Core -> worker: set busy state (not from worker itself)
    #[serde(rename = "busy")]
    Busy { id: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorKind {
    #[serde(rename = "oom")]
    Oom,
    #[serde(rename = "model")]
    Model,
    #[serde(rename = "input")]
    Input,
    #[serde(rename = "unknown")]
    Unknown,
}

/// Message from core to worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerCommand {
    Generate {
        id: String,
        prompt: String,
        sources: Vec<String>,
        mask: Option<String>,
        width: usize,
        height: usize,
        seed: i64,
        options: serde_json::Value,
        output: String,
    },
    Cancel { id: String },
}

impl WorkerCommand {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap()
    }
}

/// The worker process handle.
pub struct WorkerHandle {
    model: String,
    variant: String,
    child: Arc<Mutex<Option<Child>>>,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    state: Arc<Mutex<EngineState>>,
    device: Arc<Mutex<Option<String>>>,
    vram: Arc<Mutex<Option<f64>>>,
    shutdown: Arc<AtomicBool>,
}

impl WorkerHandle {
    /// Start the Python diffusion worker for the given model and variant.
    pub fn start(model: &str, variant: &str, models_folder: &str, compute_target: &str) -> Result<Self, String> {
        let state = Arc::new(Mutex::new(EngineState::Loading { elapsed: 0.0 }));
        let state_clone = state.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let device = Arc::new(Mutex::new(None));
        let vram = Arc::new(Mutex::new(None));

        let child = Command::new("python")
            .args(&[
                "-X", "utf8",
                "-u", // unbuffered output
                "engines/diffusion_worker.py",
                model,
                variant,
                models_folder,
                compute_target,
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .env("LATENT_DEVICE", compute_target)
            .env("HF_HOME", models_folder)
            .env("LATENT_CACHE", "") // will be set to cache dir
            .spawn()
            .map_err(|e| format!("Failed to start diffusion worker: {}", e))?;

        let mut child = child;
        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();

        // Spawn the reader task that processes worker output
        let shutdown_clone = shutdown.clone();
        let state_reader = state.clone();
        let device_reader = device.clone();
        let vram_reader = vram.clone();

        task::spawn(async move {
            use tokio::io::AsyncBufReadExt;
            // ChildStdout doesn't implement AsyncRead, use std::io::BufReader in for loop
            let reader = std::io::BufReader::new(stdout);
            for line_result in reader.lines() {
                match line_result {
                    Ok(msg_line) => {
                        let msg_line = msg_line.trim().to_string();
                        if msg_line.is_empty() { continue; }
                        if shutdown_clone.load(Ordering::Relaxed) { break; }

                        if let Ok(msg) = serde_json::from_str::<WorkerEvent>(&msg_line) {
                            match &msg {
                                WorkerEvent::Ready { model, variant, device: dev, vram: mem, .. } => {
                                    *state_reader.lock().unwrap() = EngineState::Warm;
                                    *device_reader.lock().unwrap() = Some(dev.clone());
                                    *vram_reader.lock().unwrap() = Some(*mem);
                                    log::info!("Worker ready: {} {} on {} ({} GB VRAM)", model, variant, dev, mem);
                                }
                                WorkerEvent::Done { id, output: _, seed: _, elapsed: dur, .. } => {
                                    let mut s = state_reader.lock().unwrap();
                                    *s = EngineState::Warm;
                                    log::info!("Generation done: {} in {:.1}s", id, dur);
                                }
                                WorkerEvent::Busy { .. } => {
                                    *state_reader.lock().unwrap() = EngineState::Busy;
                                }
                                WorkerEvent::Progress { step: _, steps: _, .. } => {
                                    let _s = state_reader.lock().unwrap();
                                }
                                WorkerEvent::Error { kind, message, .. } => {
                                    match kind {
                                        ErrorKind::Oom => {
                                            let mut s = state_reader.lock().unwrap();
                                            *s = EngineState::Warm;
                                            log::warn!("OOM error: {}", message);
                                        }
                                        ErrorKind::Model => {
                                            *state_reader.lock().unwrap() = EngineState::Crashed { message: message.to_string() };
                                        }
                                        _ => {}
                                    }
                                }
                                _ => {}
                            }
                            emit_event(msg);
                        } else if msg_line.starts_with("[") || msg_line.contains("WARNING") || msg_line.contains("ERROR") {
                            log::debug!("Worker log: {}", msg_line);
                        }
                    }
                    Err(_) => break,
                }
            }

            if !shutdown_clone.load(Ordering::Relaxed) {
                let mut s = state_reader.lock().unwrap();
                *s = EngineState::Crashed { message: "Worker exited unexpectedly".into() };
            }
        });

        Ok(WorkerHandle {
            model: model.to_string(),
            variant: variant.to_string(),
            child: Arc::new(Mutex::new(Some(child))),
            stdin: Arc::new(Mutex::new(Some(stdin))),
            state,
            device,
            vram,
            shutdown,
        })
    }

    /// Send a request to the worker and stream events via a channel.
    pub fn request_stream(
        &self,
        cmd: WorkerCommand,
        rx: mpsc::UnboundedSender<WorkerEvent>,
    ) -> Result<(), String> {
        // Write command to worker's stdin (sync write, worker is local process)
        let mut stdin = self.stdin.lock().unwrap();
        if let Some(stdin) = stdin.as_mut() {
            let json = serde_json::to_string(&cmd).map_err(|e| format!("Failed to serialize command: {}", e))?;
            stdin.write_all(json.as_bytes()).map_err(|e| format!("Failed to write to worker: {}", e))?;
            stdin.write_all(b"\n").map_err(|e| format!("Failed to write newline: {}", e))?;
            stdin.flush().map_err(|e| format!("Failed to flush: {}", e))?;
        } else {
            return Err("Worker stdin closed".into());
        }

        Ok(())
    }

    /// Send a cancel command to the worker.
    pub fn cancel(&self, batch_id: &str) -> Result<(), String> {
        let cmd = WorkerCommand::Cancel { id: batch_id.to_string() };
        let mut stdin = self.stdin.lock().unwrap();
        if let Some(stdin) = stdin.as_mut() {
            let json = serde_json::to_string(&cmd).map_err(|e| format!("Failed to serialize cancel: {}", e))?;
            stdin.write_all(json.as_bytes()).map_err(|e| format!("Failed to write cancel: {}", e))?;
            stdin.write_all(b"\n").map_err(|e| format!("Failed to write newline: {}", e))?;
            stdin.flush().map_err(|e| format!("Failed to flush: {}", e))?;
        }
        Ok(())
    }

    /// Unload the model from the worker.
    pub async fn unload(&self) -> Result<(), String> {
        *self.state.lock().unwrap() = EngineState::Unloading;
        let cmd = WorkerCommand::Generate {
            id: "unload".into(),
            prompt: "".into(),
            sources: vec![],
            mask: None,
            width: 1,
            height: 1,
            seed: 0,
            options: serde_json::json!({"unload": true}),
            output: "".into(),
        };
        self.request_stream(cmd, mpsc::unbounded_channel().0)?;
        Ok(())
    }

    /// Check if the worker is ready (warm).
    pub fn is_ready(&self) -> bool {
        matches!(*self.state.lock().unwrap(), EngineState::Warm)
    }

    /// Get current state.
    pub fn state(&self) -> EngineState {
        self.state.lock().unwrap().clone()
    }

    /// Get the device type.
    pub fn device(&self) -> Option<String> {
        self.device.lock().unwrap().clone()
    }

    /// Get VRAM usage.
    pub fn vram_usage(&self) -> Option<f64> {
        self.vram.lock().unwrap().clone()
    }

    /// Stop the worker.
    pub fn stop(&self) -> Result<(), String> {
        self.shutdown.store(true, Ordering::Relaxed);
        let mut child_guard = self.child.lock().unwrap();
        if let Some(ref mut child) = *child_guard {
            #[cfg(windows)]
            {
                child.kill().map_err(|e| format!("Failed to kill worker: {}", e))?;
            }
            #[cfg(not(windows))]
            {
                unsafe {
                    let pid = child.id() as libc::pid_t;
                    libc::kill(-pid, libc::SIGTERM);
                }
            }
            let _ = child.wait();
        }
        *child_guard = None;
        Ok(())
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn variant(&self) -> &str {
        &self.variant
    }
}

/// Global event channel for worker events.
static EVENT_TX: Mutex<Option<mpsc::UnboundedSender<serde_json::Value>>> = Mutex::new(None);

/// Set the global event sender (called by Tauri event system).
pub fn set_event_tx(tx: mpsc::UnboundedSender<serde_json::Value>) {
    let mut guard = EVENT_TX.lock().unwrap();
    *guard = Some(tx);
}

/// Emit an event to the UI.
fn emit_event(event: WorkerEvent) {
    if let Some(tx) = EVENT_TX.lock().unwrap().as_ref() {
        if let Ok(json) = serde_json::to_value(&event) {
            let _ = tx.send(json);
        }
    }
}
