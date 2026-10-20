//! A long-lived Python worker that speaks JSON lines on stdin/stdout (see engines/*.py).

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[cfg(windows)]
use std::os::windows::process::CommandExt;

/// Put a child in a job object that Windows closes when Latent exits, even if Latent crashes.
/// Everything the child starts joins the same job, so no engine can outlive the app and hold VRAM.
#[cfg(windows)]
pub fn adopt(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::*;
    static JOB: OnceLock<usize> = OnceLock::new();
    let job = *JOB.get_or_init(|| unsafe {
        let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            h,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        h as usize
    });
    unsafe {
        AssignProcessToJobObject(job as _, child.as_raw_handle() as _);
    }
}

/// On Linux the same guarantee comes from `hidden` (a death signal set on every child).
#[cfg(not(windows))]
pub fn adopt(_child: &Child) {}

/// Kill a process and everything it started. A venv's python.exe on Windows is a launcher that runs
/// the real interpreter as a child, so killing only the launcher would leave the model in VRAM.
#[cfg(windows)]
pub fn kill_tree(child: &mut Child) {
    let mut cmd = Command::new("taskkill");
    cmd.args(["/PID", &child.id().to_string(), "/T", "/F"]).stdout(Stdio::null()).stderr(Stdio::null());
    hidden(&mut cmd);
    let _ = cmd.status();
    let _ = child.kill();
    let _ = child.wait();
}

/// On Linux each child leads its own process group, so the whole group goes at once.
#[cfg(unix)]
pub fn kill_tree(child: &mut Child) {
    unsafe {
        libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// Every child process Latent starts goes through here.
/// Windows: no console window. Linux: its own process group (so `kill_tree` gets its children too),
/// and a death signal so it dies with Latent even if Latent crashes.
pub fn hidden(cmd: &mut Command) -> &mut Command {
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
                Ok(())
            });
        }
    }
    cmd
}

/// The interpreter inside a venv.
pub fn venv_python(env_dir: &Path) -> std::path::PathBuf {
    if cfg!(windows) {
        env_dir.join("Scripts").join("python.exe")
    } else {
        env_dir.join("bin").join("python")
    }
}

pub struct Worker {
    child: Arc<Mutex<Child>>,
    /// Shared with `sender()`, so a cancel can be written while `stream` waits for the result.
    stdin: Arc<Mutex<ChildStdin>>,
    rx: Receiver<Value>,
    pub hello: Value,
}

impl Worker {
    pub fn start(
        python: &Path,
        script: &Path,
        args: &[&str],
        envs: &[(&str, String)],
        log: &Path,
        ready_timeout: Duration,
    ) -> Result<Worker, String> {
        if !python.is_file() {
            return Err(format!("Python not found at {}", python.display()));
        }
        let log_file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log)
            .map_err(|e| format!("can't open log {}: {e}", log.display()))?;
        let mut cmd = Command::new(python);
        cmd.arg("-X").arg("utf8").arg(script).args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log_file))
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8");
        for (k, v) in envs {
            cmd.env(k, v);
        }
        hidden(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| format!("couldn't start {}: {e}", script.display()))?;
        adopt(&child);
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(v) = serde_json::from_str::<Value>(&line) {
                    if tx.send(v).is_err() {
                        break;
                    }
                }
            }
        });
        let mut worker = Worker { child: Arc::new(Mutex::new(child)), stdin: Arc::new(Mutex::new(stdin)), rx, hello: Value::Null };
        worker.hello = worker
            .wait_for("ready", ready_timeout)
            .map_err(|e| format!("{e} while loading (details in {})", log.display()))?;
        Ok(worker)
    }

    fn wait_for(&self, event: &str, timeout: Duration) -> Result<Value, String> {
        loop {
            match self.rx.recv_timeout(timeout) {
                Ok(v) => match v["event"].as_str() {
                    Some("error") => return Err(v["message"].as_str().unwrap_or("unknown error").to_string()),
                    Some(e) if e == event => return Ok(v),
                    _ => {}
                },
                Err(RecvTimeoutError::Timeout) => return Err("timed out".into()),
                Err(RecvTimeoutError::Disconnected) => return Err("the model stopped".into()),
            }
        }
    }

    /// Sends one request and passes its `progress` events to `on_progress` until it ends. Returns the final event
    /// (`done`, `cancelled` or `error`) for the caller to judge; Err only when the process stopped or went silent
    /// for `timeout`.
    pub fn stream(&self, req: &Value, timeout: Duration, mut on_progress: impl FnMut(&Value)) -> Result<Value, String> {
        send(&self.stdin, req)?;
        let id = &req["id"];
        loop {
            match self.rx.recv_timeout(timeout) {
                Ok(v) if &v["id"] != id => {}
                Ok(v) => match v["event"].as_str() {
                    Some("progress") => on_progress(&v),
                    Some("done" | "cancelled" | "error") => return Ok(v),
                    _ => {}
                },
                Err(RecvTimeoutError::Timeout) => return Err("the model didn't answer".into()),
                Err(RecvTimeoutError::Disconnected) => return Err("the model stopped".into()),
            }
        }
    }

    /// A handle that can write to the worker while a request is in flight (used by Cancel).
    pub fn sender(&self) -> Arc<Mutex<ChildStdin>> {
        self.stdin.clone()
    }

    /// A handle that can kill the process while a request is in flight (the fallback when Cancel isn't answered).
    pub fn killer(&self) -> Arc<Mutex<Child>> {
        self.child.clone()
    }
}

/// Writes one JSON line to a worker.
pub fn send(stdin: &Mutex<ChildStdin>, msg: &Value) -> Result<(), String> {
    let mut w = stdin.lock().map_err(|_| "the model stopped".to_string())?;
    writeln!(w, "{msg}").and_then(|_| w.flush()).map_err(|_| "the model stopped".to_string())
}

impl Drop for Worker {
    fn drop(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            kill_tree(&mut c);
        }
    }
}
