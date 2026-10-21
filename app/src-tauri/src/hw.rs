//! What this PC has: GPU, VRAM, CPU and RAM, and where models should run. Sent to the UI every couple of seconds.
//!
//! GPUs are found by their PCI vendor and device ids (sysfs on Linux, WMI on Windows), not by guessing from names,
//! so the choice works on any distro, with or without lspci or nvidia-smi. Each platform gathers facts into `Facts`;
//! `choose` turns them into a decision, and is the part the tests cover.

use crate::worker::hidden;
use serde::Serialize;
use std::process::Command;
use std::sync::OnceLock;
use sysinfo::System;

#[derive(Serialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
pub struct Hardware {
    /// GPU name, e.g. "NVIDIA GeForce RTX 5070 Ti" or "Intel Iris Xe Graphics". Empty if none was found.
    pub gpu: String,
    /// "nvidia" | "amd" | "intel" | ""
    pub vendor: String,
    /// Where Automatic runs models on this PC: "cuda" | "rocm" | "xpu" | "cpu".
    pub accel: String,
    /// Where the GPU setting runs them; differs from `accel` for experimental GPUs (Radeon integrated graphics).
    /// Empty when no GPU here can run models.
    pub gpu_accel: String,
    /// Why models run where they do, when that needs saying (CPU fallback, experimental support). Empty otherwise.
    pub note: String,
    /// A command that would let the GPU run models (a driver, a runtime, a group). Empty if there's nothing to fix.
    pub fix: String,
    pub vram_total: f64,
    pub vram_used: f64,
    pub cpu: String,
    pub cores: usize,
    pub ram_total: f64,
    pub ram_used: f64,
    /// Environment for models on an AMD GPU (ROCm): which GPU to use, and the compatibility target if needed.
    #[serde(skip)]
    pub rocm_env: Vec<(String, String)>,
}

const NVIDIA: u16 = 0x10de;
const AMD: u16 = 0x1002;
const INTEL: u16 = 0x8086;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// One display adapter.
#[derive(Clone, Default, Debug)]
struct Gpu {
    name: String,
    vendor: u16,
    device: u16,
    /// PCI address on Linux ("0000:03:00.0"), used to find its VRAM and its ROCm node. Empty on Windows.
    slot: String,
}

/// An AMD GPU as ROCm's kernel driver (KFD) sees it.
#[derive(Clone, Default, Debug)]
struct Kfd {
    slot: String,
    /// Instruction set, as KFD writes it: 110003 = gfx1103.
    gfx: u32,
    /// ROCm's id for the GPU ("GPU-<hex>"), used to pick it when there are several.
    uuid: String,
}

/// The distro family from /etc/os-release, used to pick install commands and package sources.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub enum Distro {
    Arch,
    Ubuntu,
    Debian,
    Fedora,
    Suse,
    #[default]
    Other,
}

/// Everything `choose` needs to know about this machine's GPU software.
#[derive(Default, Debug)]
struct Facts {
    windows: bool,
    gpus: Vec<Gpu>,
    /// The GPU name from nvidia-smi: present only when NVIDIA's driver works.
    nvidia_smi: Option<String>,
    /// NVIDIA's own kernel driver is loaded (Linux). False with nouveau or no driver.
    nvidia_driver: bool,
    kfd: Vec<Kfd>,
    /// This user can open /dev/kfd (ROCm needs it; most distros grant it through the render group).
    kfd_access: bool,
    /// Intel's GPU compute runtime (Level Zero) is installed; PyTorch's Intel GPU build needs it.
    level_zero: bool,
    /// Dedicated VRAM of each AMD GPU by slot, in GiB. Integrated graphics report only a small carve-out.
    amd_vram: Vec<(String, f64)>,
    distro: Distro,
}

#[derive(Clone, Default, Debug, PartialEq)]
struct Choice {
    gpu: String,
    vendor: String,
    accel: String,
    gpu_accel: String,
    note: String,
    fix: String,
    /// The AMD GPU whose VRAM the top bar shows (a discrete card running models).
    amd_slot: String,
    rocm_env: Vec<(String, String)>,
}

/// How PyTorch's ROCm build treats an AMD instruction set.
#[derive(Debug, PartialEq)]
enum Rocm {
    Native,
    /// Not in the build, but runs as this close relative (HSA_OVERRIDE_GFX_VERSION), in KFD's numbering.
    As(u32),
    Unsupported,
}

fn rocm_support(gfx: u32) -> Rocm {
    match (gfx / 10000, gfx / 100 % 100, gfx % 100) {
        (9, 0, 6 | 8 | 10) | (9, 4, 2) | (9, 5, 0) | (10, 3, 0) | (11, 0, 0..=2) | (11, 5, 1) | (12, 0, 0 | 1) => Rocm::Native,
        (10, 3, _) => Rocm::As(100300),     // RX 6000 cards, Radeon 680M/660M
        (11, 0 | 5, _) => Rocm::As(110000), // Radeon 780M/760M, 890M/880M
        _ => Rocm::Unsupported,
    }
}

/// "gfx1103" from KFD's 110003.
fn gfx_name(gfx: u32) -> String {
    format!("gfx{}{:x}{:x}", gfx / 10000, gfx / 100 % 100, gfx % 100)
}

/// Arc-class Intel graphics, which PyTorch's Intel GPU build supports: Arc A and B cards (0x56xx, 0xE2xx),
/// Core Ultra Meteor/Arrow Lake (0x7Dxx), Lunar Lake (0x64xx) and Panther Lake (0xB0xx). Older Iris Xe and UHD aren't.
fn intel_arc(g: &Gpu) -> bool {
    matches!(g.device >> 8, 0x56 | 0xe2 | 0x7d | 0x64 | 0xb0) || g.name.to_lowercase().contains("arc")
}

/// AMD GPUs with less dedicated memory than this are integrated graphics sharing system RAM.
const IGPU_VRAM: f64 = 3.5;

fn distro_fix(d: Distro, what: &str) -> String {
    let cmd = match (what, d) {
        ("nvidia", Distro::Arch) => "sudo pacman -S nvidia-open",
        ("nvidia", Distro::Ubuntu) => "sudo ubuntu-drivers install",
        ("nvidia", Distro::Debian) => "sudo apt install nvidia-driver",
        ("nvidia", Distro::Fedora) => "sudo dnf install akmod-nvidia",
        ("nvidia", Distro::Suse) => "sudo zypper install nvidia-video-G06",
        ("intel", Distro::Arch) => "sudo pacman -S intel-compute-runtime level-zero-loader",
        ("intel", Distro::Ubuntu | Distro::Debian) => "sudo apt install intel-opencl-icd libze1 libze-intel-gpu1",
        ("intel", Distro::Fedora) => "sudo dnf install intel-compute-runtime oneapi-level-zero",
        _ => "",
    };
    match (what, cmd.is_empty()) {
        ("nvidia", false) if d == Distro::Fedora => format!("Install NVIDIA's driver from RPM Fusion: {cmd}, then restart."),
        ("nvidia", false) if d == Distro::Suse => format!("Add NVIDIA's repository, then install its driver: {cmd}, then restart."),
        ("nvidia", false) => format!("Install NVIDIA's driver: {cmd}, then restart."),
        ("nvidia", true) => "Install NVIDIA's proprietary driver with your distribution's driver tool, then restart.".into(),
        ("intel", false) => format!("Install Intel's GPU compute runtime: {cmd}"),
        _ => "Install Intel's GPU compute runtime (Level Zero) from your distribution's packages.".into(),
    }
}

/// Picks where models run. Order: NVIDIA, then an AMD card, then Intel Arc, then Radeon integrated graphics (GPU setting only).
fn choose(f: &Facts) -> Choice {
    let by = |v: u16| f.gpus.iter().filter(move |g| g.vendor == v);
    // NVIDIA: nvidia-smi answering means the driver works.
    if let Some(name) = &f.nvidia_smi {
        return Choice { gpu: name.clone(), vendor: "nvidia".into(), accel: "cuda".into(), gpu_accel: "cuda".into(), ..Default::default() };
    }
    let nvidia = by(NVIDIA).next();
    if let Some(g) = nvidia {
        if f.nvidia_driver {
            // Driver loaded but no nvidia-smi (some distros package it separately): models run, VRAM can't be shown.
            return Choice { gpu: g.name.clone(), vendor: "nvidia".into(), accel: "cuda".into(), gpu_accel: "cuda".into(),
                note: "nvidia-smi isn't installed, so Latent can't show VRAM use. Models still run on the GPU.".into(), ..Default::default() };
        }
    }

    // AMD: ROCm, Linux only.
    let mut amd_igpu: Option<Choice> = None;
    let mut amd_note = String::new();
    let amd_gpus: Vec<&Gpu> = by(AMD).collect();
    if !amd_gpus.is_empty() && f.windows {
        amd_note = "Latent runs AMD GPUs through ROCm, which it sets up on Linux only, so models run on the CPU on Windows.".into();
    } else if !amd_gpus.is_empty() {
        let vram = |slot: &str| f.amd_vram.iter().find(|(s, _)| s == slot).map(|(_, v)| *v).unwrap_or(0.0);
        let mut cards: Vec<&Gpu> = amd_gpus.clone();
        cards.sort_by(|a, b| vram(&b.slot).total_cmp(&vram(&a.slot)));
        for g in cards {
            let Some(k) = f.kfd.iter().find(|k| k.slot == g.slot) else {
                amd_note = format!("{} has no GPU compute support in this kernel (/dev/kfd), so models run on the CPU.", g.name);
                continue;
            };
            let support = rocm_support(k.gfx);
            if support == Rocm::Unsupported {
                amd_note = format!("{} ({}) isn't supported by ROCm, so models run on the CPU.", g.name, gfx_name(k.gfx));
                continue;
            }
            if !f.kfd_access {
                return Choice { gpu: g.name.clone(), vendor: "amd".into(), accel: "cpu".into(), gpu_accel: String::new(),
                    note: "Latent can't open /dev/kfd, which ROCm needs to use the GPU, so models run on the CPU.".into(),
                    fix: "Add yourself to the render and video groups: sudo usermod -aG render,video $USER, then log out and back in.".into(),
                    ..Default::default() };
            }
            let mut env = Vec::new();
            if f.kfd.len() > 1 && !k.uuid.is_empty() {
                env.push(("ROCR_VISIBLE_DEVICES".to_string(), k.uuid.clone())); // only this GPU, not an integrated one next to it
            }
            let mut compat = String::new();
            if let Rocm::As(v) = support {
                env.push(("HSA_OVERRIDE_GFX_VERSION".to_string(), format!("{}.{}.{}", v / 10000, v / 100 % 100, v % 100)));
                compat = format!(" ROCm doesn't list {} officially, so it runs as {} (experimental).", gfx_name(k.gfx), gfx_name(v));
            }
            if vram(&g.slot) >= IGPU_VRAM {
                return Choice { gpu: g.name.clone(), vendor: "amd".into(), accel: "rocm".into(), gpu_accel: "rocm".into(),
                    note: format!("AMD GPUs run through ROCm, which is new in Latent.{compat}"), amd_slot: g.slot.clone(), rocm_env: env, ..Default::default() };
            }
            amd_igpu.get_or_insert(Choice { gpu: g.name.clone(), vendor: "amd".into(), accel: "cpu".into(), gpu_accel: "rocm".into(),
                note: "Radeon integrated graphics can run models through ROCm, but it's experimental and shares your RAM. Automatic uses the CPU; choose GPU to try it.".into(),
                rocm_env: env, ..Default::default() });
        }
    }

    // Intel: PyTorch's XPU build, Arc-class graphics only.
    if let Some(g) = by(INTEL).find(|g| intel_arc(g)) {
        if f.windows || f.level_zero {
            return Choice { gpu: g.name.clone(), vendor: "intel".into(), accel: "xpu".into(), gpu_accel: "xpu".into(), ..Default::default() };
        }
        return Choice { gpu: g.name.clone(), vendor: "intel".into(), accel: "cpu".into(), gpu_accel: "xpu".into(),
            note: "Intel's GPU compute runtime (Level Zero) isn't installed, so models run on the CPU.".into(), fix: distro_fix(f.distro, "intel"), ..Default::default() };
    }
    if let Some(c) = amd_igpu {
        return c;
    }

    // Nothing that runs models: say why.
    if let Some(g) = nvidia {
        let (note, fix) = if f.windows {
            ("NVIDIA's driver didn't answer (nvidia-smi failed), so models run on the CPU. Reinstalling the driver from nvidia.com usually fixes it.".into(), String::new())
        } else {
            ("This NVIDIA GPU needs NVIDIA's own driver to run models (the open-source nouveau driver can't), so models run on the CPU.".into(), distro_fix(f.distro, "nvidia"))
        };
        return Choice { gpu: g.name.clone(), vendor: "nvidia".into(), accel: "cpu".into(), note, fix, ..Default::default() };
    }
    if let Some(g) = amd_gpus.first() {
        return Choice { gpu: g.name.clone(), vendor: "amd".into(), accel: "cpu".into(), note: amd_note, ..Default::default() };
    }
    if let Some(g) = by(INTEL).next() {
        return Choice { gpu: g.name.clone(), vendor: "intel".into(), accel: "cpu".into(),
            note: "PyTorch's Intel GPU support starts with Arc graphics (Arc cards and Core Ultra). Older Iris Xe and UHD graphics aren't supported, so models run on the CPU.".into(),
            ..Default::default() };
    }
    let gpu = f.gpus.first().map(|g| g.name.clone()).unwrap_or_default();
    Choice { gpu, accel: "cpu".into(), ..Default::default() }
}

/// GPU and CPU identity never change while Latent runs, so they're read once.
fn fixed(sys: &mut System) -> (Choice, String, usize) {
    static FIXED: OnceLock<(Choice, String, usize)> = OnceLock::new();
    FIXED
        .get_or_init(|| {
            sys.refresh_cpu_all();
            let cpu = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
            (choose(&facts()), cpu, sys.cpus().len())
        })
        .clone()
}

pub fn snapshot(sys: &mut System) -> Hardware {
    let (c, cpu, cores) = fixed(sys);
    sys.refresh_memory();
    let (vram_total, vram_used) = match c.vendor.as_str() {
        "nvidia" => nvidia_smi().map(|(_, t, u)| (t, u)).unwrap_or_default(),
        "amd" if !c.amd_slot.is_empty() => amd_vram(&c.amd_slot),
        _ => (0.0, 0.0),
    };
    let hw = Hardware {
        gpu: c.gpu,
        vendor: c.vendor,
        accel: c.accel,
        gpu_accel: c.gpu_accel,
        note: c.note,
        fix: c.fix,
        vram_total,
        vram_used,
        cpu,
        cores,
        ram_total: sys.total_memory() as f64 / GIB,
        ram_used: sys.used_memory() as f64 / GIB,
        rocm_env: c.rocm_env,
    };
    apple_silicon(hw)
}

/// Apple Silicon Macs run models on the GPU through Metal ("mps"). CPU and GPU share one pool of memory, so the
/// memory bar shows RAM and a model's whole footprint has to fit in it (see models::fits).
#[cfg(target_os = "macos")]
fn apple_silicon(hw: Hardware) -> Hardware {
    if std::env::consts::ARCH != "aarch64" {
        return Hardware { note: "Intel Macs run models on the CPU, which is too slow for these models.".into(), ..hw };
    }
    Hardware {
        gpu: if hw.cpu.is_empty() { "Apple Silicon".into() } else { format!("{} (Metal)", hw.cpu) },
        vendor: "apple".into(),
        accel: "mps".into(),
        gpu_accel: "mps".into(),
        note: String::new(),
        fix: String::new(),
        ..hw
    }
}

#[cfg(not(target_os = "macos"))]
fn apple_silicon(hw: Hardware) -> Hardware {
    hw
}

/// Where models run, from the setting ("Automatic" | "GPU" | "CPU") and what this PC has.
pub fn resolve_device(setting: &str, hw: &Hardware) -> String {
    let pick = match setting {
        "CPU" => "",
        "GPU" if !hw.gpu_accel.is_empty() => &hw.gpu_accel,
        _ => &hw.accel,
    };
    if pick.is_empty() { "cpu".into() } else { pick.to_string() }
}

/// The PyTorch device for a place models run: ROCm's PyTorch drives AMD GPUs through the "cuda" device.
pub fn torch_device(device: &str) -> &str {
    if device == "rocm" { "cuda" } else { device }
}

/// (name, total GiB, used GiB) of the first NVIDIA GPU, or None without one.
fn nvidia_smi() -> Option<(String, f64, f64)> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"]);
    hidden(&mut cmd);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.lines().next()?.split(',').map(str::trim);
    let name = parts.next()?.to_string();
    let total: f64 = parts.next()?.parse().ok()?;
    let used: f64 = parts.next()?.parse().ok()?;
    Some((name, total / 1024.0, used / 1024.0))
}

// ---------------------------------------------------------------- Linux: sysfs

#[cfg(unix)]
fn read(path: impl AsRef<std::path::Path>) -> String {
    std::fs::read_to_string(path).map(|s| s.trim().to_string()).unwrap_or_default()
}

#[cfg(unix)]
fn hex(s: &str) -> u16 {
    u16::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

#[cfg(unix)]
fn amd_vram(slot: &str) -> (f64, f64) {
    let dev = std::path::Path::new("/sys/bus/pci/devices").join(slot);
    let gib = |f: &str| read(dev.join(f)).parse::<f64>().unwrap_or(0.0) / GIB;
    (gib("mem_info_vram_total"), gib("mem_info_vram_used"))
}

#[cfg(unix)]
fn facts() -> Facts {
    let names = lspci_names(&Command::new("lspci").arg("-D").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default());
    let mut gpus = Vec::new();
    for entry in std::fs::read_dir("/sys/bus/pci/devices").into_iter().flatten().flatten() {
        let dir = entry.path();
        if !read(dir.join("class")).starts_with("0x03") {
            continue; // display controllers only
        }
        let slot = entry.file_name().to_string_lossy().to_string();
        let (vendor, device) = (hex(&read(dir.join("vendor"))), hex(&read(dir.join("device"))));
        let fallback = match vendor { NVIDIA => "NVIDIA GPU", AMD => "AMD Radeon graphics", INTEL => "Intel graphics", _ => "Graphics adapter" };
        let name = names.iter().find(|(s, _)| *s == slot).map(|(_, n)| n.clone()).unwrap_or_else(|| format!("{fallback} ({vendor:04x}:{device:04x})"));
        gpus.push(Gpu { name, vendor, device, slot });
    }
    gpus.sort_by(|a, b| a.slot.cmp(&b.slot));

    let kfd = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes").into_iter().flatten().flatten()
        .filter_map(|node| kfd_node(&read(node.path().join("properties"))))
        .collect();
    let kfd_access = std::ffi::CString::new("/dev/kfd").map(|p| unsafe { libc::access(p.as_ptr(), libc::R_OK | libc::W_OK) } == 0).unwrap_or(false);

    // The Level Zero GPU driver, wherever this distro keeps its libraries (NixOS: /run/opengl-driver).
    let lib_dirs = ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/local/lib", "/run/opengl-driver/lib"];
    let level_zero = lib_dirs.iter().any(|d| {
        std::fs::read_dir(d).into_iter().flatten().flatten().any(|e| e.file_name().to_string_lossy().starts_with("libze_intel_gpu.so"))
    });

    let amd_vram = gpus.iter().filter(|g| g.vendor == AMD).map(|g| (g.slot.clone(), amd_vram(&g.slot).0)).collect();
    Facts {
        windows: false,
        gpus,
        nvidia_smi: nvidia_smi().map(|(n, _, _)| n),
        nvidia_driver: std::path::Path::new("/proc/driver/nvidia/version").exists(),
        kfd,
        kfd_access,
        level_zero,
        amd_vram,
        distro: distro(&read("/etc/os-release")),
    }
}

/// Readable names by PCI slot from `lspci -D`: "0000:00:02.0 VGA compatible controller: Intel Corporation ... [Iris Xe Graphics] (rev 01)".
#[cfg_attr(not(unix), allow(dead_code))]
fn lspci_names(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|l| {
            let (slot, rest) = l.split_once(' ')?;
            let name = rest.split_once(": ")?.1;
            let name = name.rsplit_once(" (rev").map(|(n, _)| n).unwrap_or(name);
            Some((slot.to_string(), name.trim().to_string()))
        })
        .collect()
}

/// A GPU from a KFD topology node's properties file ("key value" lines). CPU nodes have no SIMDs and give None.
#[cfg_attr(not(unix), allow(dead_code))]
fn kfd_node(props: &str) -> Option<Kfd> {
    let get = |k: &str| props.lines().find_map(|l| l.strip_prefix(k).and_then(|v| v.strip_prefix(' ')).and_then(|v| v.trim().parse::<u64>().ok())).unwrap_or(0);
    if get("simd_count") == 0 {
        return None;
    }
    let loc = get("location_id"); // bus << 8 | device << 3 | function
    let slot = format!("{:04x}:{:02x}:{:02x}.{:x}", get("domain"), loc >> 8, (loc >> 3) & 0x1f, loc & 7);
    let id = get("unique_id");
    Some(Kfd { slot, gfx: get("gfx_target_version") as u32, uuid: if id == 0 { String::new() } else { format!("GPU-{id:016x}") } })
}

/// The distro family from /etc/os-release, for install commands.
#[cfg_attr(not(unix), allow(dead_code))]
fn distro(os_release: &str) -> Distro {
    let field = |k: &str| os_release.lines().find_map(|l| l.strip_prefix(k)).map(|v| v.trim_matches('"').to_lowercase()).unwrap_or_default();
    let ids = format!("{} {}", field("ID="), field("ID_LIKE="));
    let has = |w: &str| ids.split_whitespace().any(|i| i == w);
    if has("arch") { Distro::Arch }
    else if has("ubuntu") { Distro::Ubuntu }
    else if has("debian") { Distro::Debian }
    else if has("fedora") { Distro::Fedora }
    else if ids.contains("suse") { Distro::Suse }
    else { Distro::Other }
}

/// The current distro family, for package choices and install commands. `Other` on non-Linux systems.
#[cfg(unix)]
pub fn current_distro() -> Distro {
    distro(&read("/etc/os-release"))
}

#[cfg(not(unix))]
pub fn current_distro() -> Distro {
    Distro::Other
}

// ---------------------------------------------------------------- Windows: WMI

#[cfg(windows)]
fn amd_vram(_slot: &str) -> (f64, f64) {
    (0.0, 0.0)
}

#[cfg(windows)]
fn facts() -> Facts {
    // "Intel(R) Arc(TM) Graphics|PCI\VEN_8086&DEV_7D55&SUBSYS_..."
    let mut cmd = Command::new("powershell");
    cmd.args(["-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }"]);
    hidden(&mut cmd);
    let text = cmd.output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
    let id = |s: &str, key: &str| s.find(key).and_then(|i| u16::from_str_radix(s.get(i + key.len()..i + key.len() + 4)?, 16).ok()).unwrap_or(0);
    let gpus = text
        .lines()
        .filter_map(|l| l.trim().split_once('|'))
        .filter(|(name, _)| !name.is_empty() && !name.contains("Microsoft Basic"))
        .map(|(name, pnp)| Gpu { name: name.to_string(), vendor: id(pnp, "VEN_"), device: id(pnp, "DEV_"), slot: String::new() })
        .collect();
    Facts { windows: true, gpus, nvidia_smi: nvidia_smi().map(|(n, _, _)| n), ..Default::default() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(name: &str, vendor: u16, device: u16, slot: &str) -> Gpu {
        Gpu { name: name.into(), vendor, device, slot: slot.into() }
    }
    fn linux(gpus: Vec<Gpu>) -> Facts {
        Facts { gpus, kfd_access: true, distro: Distro::Arch, ..Default::default() }
    }

    #[test]
    fn nvidia_with_working_driver_uses_cuda() {
        let f = Facts { nvidia_smi: Some("NVIDIA GeForce RTX 5070 Ti".into()), ..linux(vec![gpu("x", NVIDIA, 0x2c05, "0000:01:00.0")]) };
        let c = choose(&f);
        assert_eq!((c.accel.as_str(), c.gpu.as_str()), ("cuda", "NVIDIA GeForce RTX 5070 Ti"));
    }

    #[test]
    fn nvidia_without_smi_still_uses_the_gpu_if_the_driver_is_loaded() {
        let f = Facts { nvidia_driver: true, ..linux(vec![gpu("GA104", NVIDIA, 0x2484, "0000:01:00.0")]) };
        assert_eq!(choose(&f).accel, "cuda");
    }

    #[test]
    fn nvidia_on_nouveau_falls_back_with_a_driver_fix() {
        let c = choose(&linux(vec![gpu("GA104", NVIDIA, 0x2484, "0000:01:00.0")]));
        assert_eq!(c.accel, "cpu");
        assert!(c.fix.contains("pacman -S nvidia-open"), "{}", c.fix);
    }

    #[test]
    fn hybrid_laptop_prefers_nvidia_over_the_intel_igpu() {
        let f = Facts { nvidia_smi: Some("RTX 4060 Laptop".into()), ..linux(vec![gpu("Iris Xe", INTEL, 0x9a49, "0000:00:02.0"), gpu("AD107", NVIDIA, 0x28e0, "0000:01:00.0")]) };
        assert_eq!(choose(&f).accel, "cuda");
    }

    #[test]
    fn iris_xe_and_uhd_stay_on_the_cpu_and_say_why() {
        for dev in [0x9a49, 0x46a6, 0xa7a0, 0x3e92] {
            let c = choose(&Facts { level_zero: true, ..linux(vec![gpu("Intel graphics", INTEL, dev, "0000:00:02.0")]) });
            assert_eq!((c.accel.as_str(), c.gpu_accel.as_str()), ("cpu", ""));
            assert!(c.note.contains("Iris Xe"));
        }
    }

    #[test]
    fn arc_class_intel_is_found_by_device_id_even_without_arc_in_the_name() {
        for dev in [0x56a0, 0xe20b, 0x7d55, 0x7d67, 0x64a0, 0xb080] {
            let c = choose(&Facts { level_zero: true, ..linux(vec![gpu("Intel graphics (8086:xxxx)", INTEL, dev, "0000:00:02.0")]) });
            assert_eq!(c.accel, "xpu", "device {dev:04x}");
        }
    }

    #[test]
    fn arc_without_level_zero_explains_the_package_per_distro() {
        let mut f = linux(vec![gpu("Arc A770", INTEL, 0x56a0, "0000:03:00.0")]);
        f.distro = Distro::Ubuntu;
        let c = choose(&f);
        assert_eq!((c.accel.as_str(), c.gpu_accel.as_str()), ("cpu", "xpu"));
        assert!(c.fix.contains("apt install"), "{}", c.fix);
        f.distro = Distro::Other;
        assert!(choose(&f).fix.contains("Level Zero"));
    }

    fn radeon(gfx: u32, vram: f64) -> Facts {
        let slot = "0000:03:00.0";
        Facts { kfd: vec![Kfd { slot: slot.into(), gfx, uuid: "GPU-1".into() }], amd_vram: vec![(slot.into(), vram)], ..linux(vec![gpu("Radeon", AMD, 0x744c, slot)]) }
    }

    #[test]
    fn supported_radeon_card_uses_rocm_without_overrides() {
        let c = choose(&radeon(110000, 24.0));
        assert_eq!(c.accel, "rocm");
        assert!(c.rocm_env.is_empty(), "{:?}", c.rocm_env);
        assert_eq!(c.amd_slot, "0000:03:00.0");
    }

    #[test]
    fn rx_6700_runs_as_gfx1030() {
        let c = choose(&radeon(100301, 12.0));
        assert_eq!(c.accel, "rocm");
        assert_eq!(c.rocm_env, vec![("HSA_OVERRIDE_GFX_VERSION".to_string(), "10.3.0".to_string())]);
        assert!(c.note.contains("gfx1031") && c.note.contains("as gfx1030"), "{}", c.note);
    }

    #[test]
    fn radeon_780m_is_gpu_setting_only() {
        let c = choose(&radeon(110003, 0.5));
        assert_eq!((c.accel.as_str(), c.gpu_accel.as_str()), ("cpu", "rocm"));
        assert_eq!(c.rocm_env, vec![("HSA_OVERRIDE_GFX_VERSION".to_string(), "11.0.0".to_string())]);
    }

    #[test]
    fn rdna1_and_vega_are_unsupported() {
        assert_eq!(rocm_support(90000), Rocm::Unsupported);
        let c = choose(&radeon(100100, 8.0));
        assert_eq!((c.accel.as_str(), c.gpu_accel.as_str()), ("cpu", ""));
        assert!(c.note.contains("gfx1010"), "{}", c.note);
    }

    #[test]
    fn no_kfd_access_asks_for_the_render_group() {
        let mut f = radeon(110000, 24.0);
        f.kfd_access = false;
        let c = choose(&f);
        assert_eq!(c.accel, "cpu");
        assert!(c.fix.contains("usermod -aG render,video"));
    }

    #[test]
    fn ryzen_desktop_igpu_next_to_a_radeon_card_picks_the_card_only() {
        let (card, igpu) = ("0000:03:00.0", "0000:7c:00.0");
        let f = Facts {
            kfd: vec![Kfd { slot: card.into(), gfx: 110000, uuid: "GPU-aaaa".into() }, Kfd { slot: igpu.into(), gfx: 100306, uuid: "GPU-bbbb".into() }],
            amd_vram: vec![(card.into(), 24.0), (igpu.into(), 0.5)],
            ..linux(vec![gpu("Raphael", AMD, 0x164e, igpu), gpu("Navi 31", AMD, 0x744c, card)])
        };
        let c = choose(&f);
        assert_eq!((c.accel.as_str(), c.gpu.as_str()), ("rocm", "Navi 31"));
        assert_eq!(c.rocm_env, vec![("ROCR_VISIBLE_DEVICES".to_string(), "GPU-aaaa".to_string())]);
    }

    #[test]
    fn amd_on_windows_says_rocm_is_linux_only() {
        let f = Facts { windows: true, gpus: vec![gpu("AMD Radeon RX 7900 XTX", AMD, 0x744c, "")], ..Default::default() };
        let c = choose(&f);
        assert_eq!(c.accel, "cpu");
        assert!(c.note.contains("Linux only"));
    }

    #[test]
    fn devices_resolve_from_the_setting() {
        let hw = Hardware { accel: "cpu".into(), gpu_accel: "rocm".into(), ..Default::default() };
        assert_eq!(resolve_device("Automatic", &hw), "cpu");
        assert_eq!(resolve_device("GPU", &hw), "rocm");
        assert_eq!(resolve_device("CPU", &Hardware { accel: "cuda".into(), ..Default::default() }), "cpu");
        assert_eq!(torch_device("rocm"), "cuda");
    }

    #[test]
    fn parses_lspci_and_kfd_output() {
        let names = lspci_names("0000:00:02.0 VGA compatible controller: Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics] (rev 01)
            0000:03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] (rev c8)
");
        assert_eq!(names[0], ("0000:00:02.0".to_string(), "Intel Corporation TigerLake-LP GT2 [Iris Xe Graphics]".to_string()));
        assert!(names[1].1.ends_with("[Radeon RX 7900 XT/7900 XTX/7900M]"));
        // A real KFD node: location_id 768 = bus 03, device 00, function 0.
        let gpu = kfd_node("cpu_cores_count 0
simd_count 192
mem_banks_count 1
gfx_target_version 110000
location_id 768
domain 0
unique_id 12345678901234
").unwrap();
        assert_eq!((gpu.slot.as_str(), gpu.gfx, gpu.uuid.as_str()), ("0000:03:00.0", 110000, "GPU-00000b3a73ce2ff2"));
        assert!(kfd_node("cpu_cores_count 16
simd_count 0
").is_none());
        let apu = kfd_node("simd_count 24
gfx_target_version 110003
location_id 50176
domain 0
").unwrap();
        assert_eq!((apu.slot.as_str(), apu.uuid.as_str()), ("0000:c4:00.0", ""));
    }

    #[test]
    fn distro_families() {
        assert_eq!(distro("ID=cachyos\nID_LIKE=arch\n"), Distro::Arch);
        assert_eq!(distro("ID=pop\nID_LIKE=\"ubuntu debian\"\n"), Distro::Ubuntu);
        assert_eq!(distro("ID=debian\n"), Distro::Debian);
        assert_eq!(distro("ID=nobara\nID_LIKE=\"rhel centos fedora\"\n"), Distro::Fedora);
        assert_eq!(distro("ID=\"opensuse-tumbleweed\"\nID_LIKE=\"opensuse suse\"\n"), Distro::Suse);
        assert_eq!(distro("ID=nixos\n"), Distro::Other);
        assert_eq!(gfx_name(110003), "gfx1103");
        assert_eq!(gfx_name(90010), "gfx90a");
    }
}
