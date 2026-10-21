//! Remote access: Latent's own interface served over HTTP, so a browser on another device can use this PC's GPU.
//! Two ways in: this network (a LAN address), or anywhere through Tailscale (`tailscale serve` adds HTTPS). Off until
//! the user starts it in Settings. On this network traffic is plain HTTP, so the secret and cookie can be sniffed on
//! that Wi-Fi; the UI says so, and points to Tailscale.
//!
//! Every request needs the secret from the pairing link: /pair?t=<secret> stores it as a cookie, and the browser
//! sends it with everything after (fetch, the event stream, images). "New link" changes the secret, which signs out
//! every device that used the old one.
//!
//! Requests: GET /api/events (server-sent events: the same events the window gets), POST /api/<command> (the
//! same commands, see `remote_call` in main.rs), GET /api/file (show a frame or a source), GET /api/download (save a
//! frame to the browser's device), POST /api/upload (import a photo from it). Everything else is the UI itself.
//! Same design as Mimic's remote.rs.

use crate::{App, Shared};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Listener, Manager};

/// Not Mimic's 4777 to 4786, so both apps can share at once.
const PORTS: std::ops::Range<u16> = 4877..4887;
/// Tailscale serves HTTPS on 443 by default, which Mimic takes; Latent uses another port Tailscale allows.
const TAILSCALE_PORT: u16 = 8443;
const MAX_BODY: usize = 64 * 1024 * 1024; // a photo from a phone, or a pasted image as base64
/// Connections served at once (each browser holds one for events, plus a few for requests); the rest wait.
const MAX_CONNECTIONS: usize = 32;
/// Sent with every response: no sniffing, no framing by other sites, and the pairing link never leaks as a referrer.
const SECURITY_HEADERS: &str = "X-Content-Type-Options: nosniff\r\nX-Frame-Options: DENY\r\nReferrer-Policy: no-referrer\r\n";
/// The page's own policy, the same as the window's (tauri.conf.json): only Latent's own scripts run.
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";
/// Events the window gets that remote browsers get too (not "update-progress": updates are the host's).
const EVENTS: &[&str] = &["hw", "engine", "progress", "frame", "batch", "install", "log", "caelestia", "fallback"];

#[derive(Default)]
pub struct Remote {
    running: Mutex<Option<Running>>,
    /// Event streams of connected browsers.
    clients: Mutex<Vec<Sender<String>>>,
}

struct Running {
    mode: String,
    /// The address the server listens on: the home-network adapter, or localhost behind Tailscale.
    host: IpAddr,
    port: u16,
    /// The link without the secret, e.g. https://desktop.tail1234.ts.net:8443
    base: String,
    stop: Arc<AtomicBool>,
}

impl Remote {
    /// Forwards the window's events to connected browsers. Called once at startup; costs nothing while sharing is off.
    pub fn init(handle: &AppHandle) {
        for name in EVENTS {
            let h = handle.clone();
            handle.listen_any(*name, move |e| {
                let app = h.state::<Shared>();
                let msg = format!("event: {name}\ndata: {}\n\n", e.payload());
                app.remote.clients.lock().unwrap().retain(|c| c.send(msg.clone()).is_ok());
            });
        }
    }

    pub fn status(&self, app: &App) -> Value {
        let r = self.running.lock().unwrap();
        let devices = self.clients.lock().unwrap().len();
        match r.as_ref() {
            Some(r) => json!({ "on": true, "mode": r.mode, "url": format!("{}/pair?t={}", r.base, secret(app)), "devices": devices }),
            None => json!({ "on": false }),
        }
    }
}

/// Start sharing: "lan" (this network) or "tailscale" (anywhere, HTTPS).
pub fn start(handle: &AppHandle, mode: &str) -> Result<Value, String> {
    let app = handle.state::<Shared>();
    stop(handle);
    // On this network the server listens on the home-network adapter only (never every interface, so a PC that is
    // also plugged straight into the internet stays closed there); through Tailscale only Tailscale's proxy reaches it.
    let host = match mode {
        "lan" => lan_ip()?,
        "tailscale" => IpAddr::V4(Ipv4Addr::LOCALHOST),
        other => return Err(format!("unknown sharing mode {other}")),
    };
    let (listener, port) = PORTS
        .clone()
        .find_map(|p| TcpListener::bind((host, p)).ok().map(|l| (l, p)))
        .ok_or("Ports 4877 to 4886 are all in use, so Latent can't share.")?;
    let base = if mode == "lan" { format!("http://{host}:{port}") } else { tailscale_serve(port)? };
    let stop = Arc::new(AtomicBool::new(false));
    let (h, flag) = (handle.clone(), stop.clone());
    std::thread::spawn(move || {
        let busy = Arc::new(AtomicUsize::new(0));
        for conn in listener.incoming() {
            if flag.load(Ordering::SeqCst) {
                break;
            }
            let Ok(conn) = conn else { continue };
            while busy.load(Ordering::SeqCst) >= MAX_CONNECTIONS {
                std::thread::sleep(Duration::from_millis(50));
            }
            busy.fetch_add(1, Ordering::SeqCst);
            let (h, busy) = (h.clone(), busy.clone());
            std::thread::spawn(move || {
                let _ = serve(&h, conn);
                busy.fetch_sub(1, Ordering::SeqCst);
            });
        }
    });
    *app.remote.running.lock().unwrap() = Some(Running { mode: mode.into(), host, port, base, stop });
    let status = app.remote.status(&app);
    let _ = handle.emit("remote", &status);
    Ok(status)
}

/// Stop sharing: closes the port, ends every browser's session and removes Latent's Tailscale route.
pub fn stop(handle: &AppHandle) -> Value {
    let app = handle.state::<Shared>();
    let running = app.remote.running.lock().unwrap().take(); // its own statement: the lock is released before status() takes it again
    let Some(r) = running else { return app.remote.status(&app) };
    r.stop.store(true, Ordering::SeqCst);
    let _ = TcpStream::connect((r.host, r.port)); // wakes the accept loop so it sees the flag
    app.remote.clients.lock().unwrap().clear();
    if r.mode == "tailscale" {
        if let Some(ts) = tailscale() {
            let _ = run(Command::new(ts).args(["serve", &format!("--https={TAILSCALE_PORT}"), "off"]), Duration::from_secs(10));
        }
    }
    let status = app.remote.status(&app);
    let _ = handle.emit("remote", &status);
    status
}

/// A new secret: old links and every signed-in browser stop working.
pub fn new_link(handle: &AppHandle) -> Result<Value, String> {
    let app = handle.state::<Shared>();
    std::fs::write(secret_file(&app), fresh_secret()).map_err(|e| e.to_string())?;
    crate::store::private(&secret_file(&app));
    app.remote.clients.lock().unwrap().clear();
    let status = app.remote.status(&app);
    let _ = handle.emit("remote", &status);
    Ok(status)
}

// ---------------------------------------------------------------- the secret

fn secret_file(app: &App) -> PathBuf {
    app.dirs.settings.with_file_name("remote-secret")
}

fn fresh_secret() -> String {
    let mut b = [0u8; 24];
    getrandom::getrandom(&mut b).expect("the OS random number generator failed");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// The pairing secret, made the first time it's needed and kept until "New link".
fn secret(app: &App) -> String {
    let path = secret_file(app);
    if let Some(s) = std::fs::read_to_string(&path).ok().map(|s| s.trim().to_string()).filter(|s| s.len() >= 32) {
        return s;
    }
    let s = fresh_secret();
    let _ = std::fs::write(&path, &s);
    crate::store::private(&path);
    s
}

/// Compares in constant time, so response timing doesn't reveal the secret.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

// ---------------------------------------------------------------- networks

/// This PC's address on a home or office network, whichever adapter has it (a PC can be on Wi-Fi at home and
/// plugged straight into the internet at the same time). Only a private address will do: a public one would put
/// Latent on the whole internet over plain HTTP.
fn lan_ip() -> Result<IpAddr, String> {
    let up: Vec<(String, IpAddr)> = if_addrs::get_if_addrs().map_err(|e| e.to_string())?.into_iter().filter(|i| i.is_oper_up()).map(|i| (i.name.clone(), i.ip())).collect();
    pick_lan(&up).ok_or_else(|| {
        "This PC isn't on a home or office network (it's either offline or connected straight to the internet), so there's nothing safe to share on. Use Anywhere (Tailscale) instead: it's encrypted and only your devices can reach it.".into()
    })
}

/// Adapters that aren't a real network: WSL, Docker and VM switches, VPNs and Tailscale.
const VIRTUAL: &[&str] = &["vethernet", "wsl", "hyper-v", "docker", "virbr", "br-", "veth", "vmware", "virtualbox", "vbox", "tailscale", "zerotier", "wireguard", "tun", "tap", "loopback"];

/// The home-network address among this PC's adapters: a private IPv4 (192.168.x.x first, then 10.x, then
/// 172.16–31.x) on a real adapter. None if there's no such adapter.
fn pick_lan(adapters: &[(String, IpAddr)]) -> Option<IpAddr> {
    adapters
        .iter()
        .filter_map(|(name, ip)| match ip {
            IpAddr::V4(v) if v.is_private() && !VIRTUAL.iter().any(|w| name.to_lowercase().contains(w)) => Some(*v),
            _ => None,
        })
        .min_by_key(|v| match v.octets() {
            [192, 168, ..] => 0,
            [10, ..] => 1,
            _ => 2,
        })
        .map(IpAddr::V4)
}

fn tailscale() -> Option<PathBuf> {
    let installed = PathBuf::from(r"C:\Program Files\Tailscale\tailscale.exe");
    if cfg!(windows) && installed.is_file() {
        return Some(installed);
    }
    let mut probe = Command::new("tailscale");
    probe.arg("version").stdout(Stdio::null()).stderr(Stdio::null());
    crate::worker::hidden(&mut probe);
    probe.status().ok().filter(|s| s.success()).map(|_| PathBuf::from("tailscale"))
}

/// Runs a command, giving up after `limit` (`tailscale serve` waits forever when HTTPS isn't enabled yet).
fn run(cmd: &mut Command, limit: Duration) -> Result<String, String> {
    crate::worker::hidden(cmd);
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null()).spawn().map_err(|e| e.to_string())?;
    let (mut out, mut err) = (child.stdout.take().unwrap(), child.stderr.take().unwrap());
    let reader = std::thread::spawn(move || {
        let (mut a, mut b) = (String::new(), String::new());
        let _ = out.read_to_string(&mut a);
        let _ = err.read_to_string(&mut b);
        a + &b
    });
    let t0 = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            break Some(s);
        }
        if t0.elapsed() > limit {
            crate::worker::kill_tree(&mut child);
            break None;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let text = reader.join().unwrap_or_default();
    match status {
        Some(s) if s.success() => Ok(text),
        _ => Err(text),
    }
}

/// Puts this PC's port behind Tailscale's HTTPS, and returns its address on the tailnet.
fn tailscale_serve(port: u16) -> Result<String, String> {
    let ts = tailscale().ok_or("Tailscale isn't installed on this PC. https://tailscale.com/download Then sign in on this PC and on your other device with the same account, and press Anywhere (Tailscale) again.")?;
    let status: Value = serde_json::from_str(&run(Command::new(&ts).args(["status", "--json"]), Duration::from_secs(10)).unwrap_or_default()).unwrap_or_default();
    if status["BackendState"].as_str() != Some("Running") {
        return Err("Tailscale isn't connected. Open Tailscale and sign in, then try again.".into());
    }
    let name = status["Self"]["DNSName"].as_str().unwrap_or("").trim_end_matches('.').to_string();
    if name.is_empty() {
        return Err("Tailscale has no name for this PC. Turn on MagicDNS in the Tailscale admin console (DNS page), then try again.".into());
    }
    let target = format!("http://127.0.0.1:{port}");
    run(Command::new(&ts).args(["serve", "--bg", &format!("--https={TAILSCALE_PORT}"), &target]), Duration::from_secs(20)).map_err(|out| {
        let link = out.split_whitespace().find(|w| w.starts_with("https://login.tailscale.com")).map(String::from);
        if let Some(link) = link {
            format!("Tailscale needs HTTPS turned on for your devices first (once). {link} Then press Anywhere (Tailscale) again.")
        } else if out.contains("Access denied") || out.contains("permission") {
            "Tailscale won't let Latent set up sharing. Run: sudo tailscale set --operator=$USER, then try again.".into()
        } else {
            format!("Tailscale couldn't share Latent: {}", out.trim())
        }
    })?;
    Ok(format!("https://{name}:{TAILSCALE_PORT}"))
}

// ---------------------------------------------------------------- HTTP

struct Req {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Req {
    fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(h, _)| h == k).map(|(_, v)| v.as_str())
    }
    fn q(&self, k: &str) -> String {
        self.query.iter().find(|(h, _)| h == k).map(|(_, v)| v.clone()).unwrap_or_default()
    }
    fn cookie(&self, k: &str) -> Option<String> {
        self.header("cookie")?.split(';').filter_map(|c| c.trim().split_once('=')).find(|(n, _)| *n == k).map(|(_, v)| v.to_string())
    }
}

fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < b.len() => match u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16) {
                Ok(v) => {
                    out.push(v);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            c => out.push(c),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The request line and headers. The body is read separately, once the request is known to be allowed.
fn read_head(conn: &TcpStream) -> std::io::Result<(Req, BufReader<TcpStream>)> {
    let bad = |m: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, m.to_string());
    let mut r = BufReader::new(conn.try_clone()?);
    let mut line = String::new();
    (&mut r).take(8192).read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, target) = (parts.next().ok_or_else(|| bad("empty request"))?.to_string(), parts.next().ok_or_else(|| bad("no target"))?.to_string());
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        (&mut r).take(16384).read_line(&mut h)?;
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if headers.len() > 100 {
            return Err(bad("too many headers"));
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }
    let (path, qs) = target.split_once('?').unwrap_or((&target, ""));
    let query = qs.split('&').filter(|p| !p.is_empty()).map(|p| p.split_once('=').unwrap_or((p, ""))).map(|(k, v)| (decode(k), decode(v))).collect();
    Ok((Req { method, path: decode(path), query, headers, body: Vec::new() }, r))
}

fn read_body(req: &mut Req, r: &mut BufReader<TcpStream>) -> std::io::Result<()> {
    let len: usize = req.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if len > MAX_BODY {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "request too large"));
    }
    req.body = vec![0; len];
    r.read_exact(&mut req.body)
}

fn respond(conn: &mut TcpStream, status: &str, ctype: &str, body: &[u8], extra: &[(&str, String)]) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n{SECURITY_HEADERS}", body.len());
    for (k, v) in extra {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    conn.write_all(head.as_bytes())?;
    conn.write_all(body)?;
    conn.flush()
}

fn json_reply(conn: &mut TcpStream, r: Result<Value, String>) -> std::io::Result<()> {
    match r {
        Ok(v) => respond(conn, "200 OK", "application/json", json!({ "ok": v }).to_string().as_bytes(), &[]),
        Err(e) => respond(conn, "400 Bad Request", "application/json", json!({ "error": e }).to_string().as_bytes(), &[]),
    }
}

const LOCKED: &str = "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>Latent</title>\
<body style='font:16px system-ui;background:#111;color:#ddd;display:grid;place-items:center;height:90vh;text-align:center'>\
<div><h2>Open the link from Latent</h2><p>On the PC running Latent, open Settings › Use from another device and copy the link.</p></div>";

fn serve(handle: &AppHandle, mut conn: TcpStream) -> std::io::Result<()> {
    conn.set_read_timeout(Some(Duration::from_secs(60)))?;
    let (mut req, mut reader) = read_head(&conn)?;
    let app = handle.state::<Shared>();
    let key = secret(&app);
    let tls = app.remote.running.lock().unwrap().as_ref().map(|r| r.mode == "tailscale").unwrap_or(false);

    // Pairing: the secret from the link becomes a cookie, then the address bar is cleaned up.
    if req.path == "/pair" {
        if !same(&req.q("t"), &key) {
            return respond(&mut conn, "403 Forbidden", "text/html; charset=utf-8", LOCKED.as_bytes(), &[]);
        }
        let cookie = format!("latent={key}; Path=/; Max-Age=31536000; HttpOnly; SameSite=Strict{}", if tls { "; Secure" } else { "" });
        return respond(&mut conn, "303 See Other", "text/plain", b"", &[("Location", "/".into()), ("Set-Cookie", cookie)]);
    }
    if !req.cookie("latent").map(|c| same(&c, &key)).unwrap_or(false) {
        return respond(&mut conn, "401 Unauthorized", "text/html; charset=utf-8", LOCKED.as_bytes(), &[]);
    }
    // Writes must come from Latent's own page, not another site the browser has open.
    if req.method == "POST" {
        if let (Some(origin), Some(host)) = (req.header("origin"), req.header("host")) {
            if origin.split("://").nth(1) != Some(host) {
                return respond(&mut conn, "403 Forbidden", "text/plain", b"cross-site request", &[]);
            }
        }
        read_body(&mut req, &mut reader)?;
    }

    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/api/events") => events(&app, conn),
        ("GET", "/api/file") => file(&app, &mut conn, &req),
        ("GET", "/api/download") => download(&app, &mut conn, &req),
        ("POST", "/api/upload") => {
            let r = upload(&app, &req);
            json_reply(&mut conn, r)
        }
        ("POST", p) if p.starts_with("/api/") => {
            let args: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
            let r = crate::remote_call(handle, &p[5..], args);
            json_reply(&mut conn, r)
        }
        ("GET", p) => {
            let name = if p == "/" { "index.html" } else { p.trim_start_matches('/') };
            match handle.asset_resolver().get(name.to_string()) {
                Some(a) => respond(&mut conn, "200 OK", &a.mime_type, &a.bytes, &[("Content-Security-Policy", CSP.into())]),
                None => respond(&mut conn, "404 Not Found", "text/plain", b"not found", &[]),
            }
        }
        _ => respond(&mut conn, "405 Method Not Allowed", "text/plain", b"", &[]),
    }
}

/// The window's events, as server-sent events, until the browser goes away or sharing stops.
fn events(app: &App, mut conn: TcpStream) -> std::io::Result<()> {
    conn.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\nX-Accel-Buffering: no\r\n{SECURITY_HEADERS}\r\n: hello\n\n").as_bytes())?;
    conn.flush()?;
    let (tx, rx) = channel();
    app.remote.clients.lock().unwrap().push(tx);
    loop {
        let msg = match rx.recv_timeout(Duration::from_secs(15)) {
            Ok(m) => m,
            Err(RecvTimeoutError::Timeout) => ": ping\n\n".into(), // keeps proxies from closing a quiet stream
            Err(RecvTimeoutError::Disconnected) => return Ok(()),
        };
        conn.write_all(msg.as_bytes())?;
        conn.flush()?;
    }
}

/// Paths a browser may read: Latent's own frames, previews and sources. Never anything else on this PC.
pub fn check_path(app: &App, path: &str) -> Result<PathBuf, String> {
    let p = Path::new(path).canonicalize().map_err(|_| "That file no longer exists.".to_string())?;
    if app.library.serves(&p) {
        Ok(p)
    } else {
        Err("Latent only shares its own frames and sources.".into())
    }
}

fn image_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "application/octet-stream",
    }
}

/// Shows a frame, a preview or a source.
fn file(app: &App, conn: &mut TcpStream, req: &Req) -> std::io::Result<()> {
    let p = match check_path(app, &req.q("path")) {
        Ok(p) => p,
        Err(e) => return respond(conn, "404 Not Found", "text/plain", e.as_bytes(), &[]),
    };
    respond(conn, "200 OK", image_type(&p), &std::fs::read(&p)?, &[])
}

/// A file name that is safe in a header and on any file system.
fn clean_name(name: &str) -> String {
    name.chars().filter(|c| !"\\/:*?\"<>|\r\n".contains(*c)).collect()
}

/// Saves a frame on the browser's device: exported in the Settings format, sent as a file download. Asked for by
/// frame id, so the browser never names a file.
fn download(app: &App, conn: &mut TcpStream, req: &Req) -> std::io::Result<()> {
    let frame = match app.library.frame(&req.q("id")) {
        Ok(f) => f,
        Err(e) => return respond(conn, "404 Not Found", "text/plain", e.as_bytes(), &[]),
    };
    let ext = if app.settings.lock().unwrap().format == "JPEG" { "jpg" } else { "png" };
    let tmp = app.library.frames_dir.join(format!("download-{}.{ext}", crate::library::now_ms()));
    let result = crate::export(&frame.path, &tmp).and_then(|_| std::fs::read(&tmp).map_err(|e| e.to_string()));
    let _ = std::fs::remove_file(&tmp);
    let data = match result {
        Ok(d) => d,
        Err(e) => return respond(conn, "500 Internal Server Error", "text/plain", e.as_bytes(), &[]),
    };
    let name = clean_name(&crate::file_stem(frame.recipe["prompt"].as_str().unwrap_or_default(), frame.seed));
    let ascii: String = name.chars().map(|c| if c.is_ascii() { c } else { '_' }).collect();
    let utf8: String = format!("{name}.{ext}").bytes().map(|b| if b.is_ascii_alphanumeric() || b"-._".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect();
    let disposition = format!("attachment; filename=\"{ascii}.{ext}\"; filename*=UTF-8''{utf8}");
    respond(conn, "200 OK", image_type(&tmp), &data, &[("Content-Disposition", disposition)])
}

/// Imports a photo sent from the browser's device as a source.
fn upload(app: &App, req: &Req) -> Result<Value, String> {
    if req.body.is_empty() {
        return Err("The file was empty.".into());
    }
    // Keep the file's own name, so the source is named after it.
    let name = clean_name(&req.q("name"));
    let name = if name.trim().trim_matches('.').is_empty() { "Uploaded photo.png".to_string() } else { name };
    let dir = app.library.frames_dir.join(format!("upload-{}", crate::library::now_ms()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let src = dir.join(&name);
    let result = std::fs::write(&src, &req.body).map_err(|e| e.to_string()).and_then(|_| app.library.import(&src));
    let _ = std::fs::remove_dir_all(&dir);
    serde_json::to_value(result?).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_compare_exactly() {
        assert!(same("abc123", "abc123"));
        assert!(!same("abc123", "abc124"));
        assert!(!same("abc", "abc123"));
        assert_eq!(fresh_secret().len(), 48);
        assert_ne!(fresh_secret(), fresh_secret());
    }

    fn adapters(list: &[(&str, &str)]) -> Vec<(String, IpAddr)> {
        list.iter().map(|(n, ip)| (n.to_string(), ip.parse().unwrap())).collect()
    }

    #[test]
    fn shares_on_the_home_wifi_not_the_public_ethernet_or_wsl() {
        let a = adapters(&[("Ethernet", "64.61.233.101"), ("Wi-Fi", "192.168.0.127"), ("vEthernet (WSL (Hyper-V firewall))", "172.19.192.1")]);
        assert_eq!(pick_lan(&a), Some("192.168.0.127".parse().unwrap()));
    }

    #[test]
    fn no_home_network_means_no_lan_sharing() {
        assert_eq!(pick_lan(&adapters(&[("Ethernet", "64.61.233.101"), ("Loopback Pseudo-Interface 1", "127.0.0.1")])), None);
        assert_eq!(pick_lan(&adapters(&[("tailscale0", "100.101.2.3"), ("docker0", "172.17.0.1")])), None);
    }

    #[test]
    fn linux_adapters() {
        assert_eq!(pick_lan(&adapters(&[("lo", "127.0.0.1"), ("tailscale0", "100.101.2.3"), ("enp3s0", "10.0.0.5")])), Some("10.0.0.5".parse().unwrap()));
    }

    #[test]
    fn decodes_query_strings() {
        assert_eq!(decode("C%3A%5CUsers%5Cmason%5Cframe%201.png"), r"C:\Users\mason\frame 1.png");
        assert_eq!(decode("a+b%2"), "a b%2");
        assert_eq!(decode("%E2%9C%93"), "✓");
    }

    #[test]
    fn file_names_lose_what_a_header_or_disk_cannot_hold() {
        assert_eq!(clean_name("../a\"b\r\n.png"), "..ab.png");
        assert_eq!(clean_name("lighthouse at dusk"), "lighthouse at dusk");
    }
}
