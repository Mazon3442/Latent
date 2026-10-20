/// Hardware detection: GPU, VRAM, RAM, disk space.
/// Adapted from Mimic's hw.rs.
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct HwInfo {
    pub devices: Vec<GpuDevice>,
    pub ram_total: u64,
    pub ram_used: u64,
    pub disk_free: u64,
    pub os: String,
    pub cpu: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuDevice {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_total_mb: Option<u64>,
    pub vram_used_mb: Option<u64>,
    pub driver: Option<String>,
    pub cuda_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Cpu,
}

impl std::fmt::Display for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

pub fn detect() -> HwInfo {
    let mut devices = Vec::new();
    let mut ram_total: u64 = 0;
    let mut ram_used: u64 = 0;
    let mut disk_free: u64 = 0;

    #[cfg(windows)]
    {
        devices.extend(detect_nvidia());
        devices.extend(detect_amd_windows());
        devices.extend(detect_intel_windows());
        ram_total = windows_ram_total();
        ram_used = windows_ram_used();
        disk_free = windows_disk_free();
    }

    #[cfg(target_os = "linux")]
    {
        devices.extend(detect_nvidia_linux());
        devices.extend(detect_amd_linux());
        devices.extend(detect_intel_linux());
        ram_total = linux_ram_total();
        ram_used = linux_ram_used();
        disk_free = linux_disk_free();
    }

    // Always include CPU as fallback
    let cpu_name = detect_cpu();
    if devices.is_empty() || devices.iter().all(|d| d.vram_total_mb.unwrap_or(0) < 2048) {
        devices.push(GpuDevice {
            vendor: GpuVendor::Cpu,
            name: cpu_name.clone(),
            vram_total_mb: None,
            vram_used_mb: None,
            driver: Some("cpu".into()),
            cuda_version: None,
        });
    }

    HwInfo {
        devices,
        ram_total,
        ram_used,
        disk_free,
        os: std::env::consts::OS.to_string(),
        cpu: cpu_name,
    }
}

// ── NVIDIA ──────────────────────────────────────────────────────

#[cfg(windows)]
fn detect_nvidia() -> Vec<GpuDevice> {
    // Try NVML first via nvml-wrapper
    if let Ok(mut nvml) = nvml_wrapper::Nvml::init() {
        let mut out = Vec::new();
        if let Ok(device_count) = nvml.device_count() {
            for i in 0..device_count {
                if let Ok(device) = nvml.device_by_index(i) {
                    let name = device.name().unwrap_or_else(|_| "NVIDIA GPU".to_string());
                    let vram = device.memory_info().ok().map(|m| m.total);
                    let used = device.memory_info().ok().map(|m| m.used);
                    out.push(GpuDevice {
                        vendor: GpuVendor::Nvidia,
                        name,
                        vram_total_mb: vram.map(|v| v / 1024 / 1024),
                        vram_used_mb: used.map(|u| u / 1024 / 1024),
                        driver: device.firmware_version().ok(),
                    });
                }
            }
        }
        return out;
    }

    // Fallback: scan for NVIDIA devices via registry
    // Simple heuristic: look for NVIDIA in device names
    let mut out = Vec::new();
    if let Some(info) = nvidia_fallback_info() {
        out.push(GpuDevice {
            vendor: GpuVendor::Nvidia,
            name: info.0,
            vram_total_mb: Some(info.1),
            vram_used_mb: None,
            driver: None,
            cuda_version: None,
        });
    }
    out
}

#[cfg(windows)]
fn cuda_version_from_driver(driver: Option<&str>) -> Option<String> {
    // Parse CUDA version from driver version string
    // NVIDIA Windows driver versioning: 5xx.xx = supports CUDA 12.x
    driver
        .map(|d| {
            let major = d
                .split('.')
                .next()
                .unwrap_or("0")
                .parse::<u32>()
                .unwrap_or(0);
            // Rough mapping: 52x = CUDA 11.x, 53x-55x = CUDA 12.x
            if major >= 550 {
                "12.8"
            } else if major >= 545 {
                "12.6"
            } else if major >= 540 {
                "12.4"
            } else if major >= 535 {
                "12.2"
            } else if major >= 530 {
                "12.1"
            } else if major >= 525 {
                "12.0"
            } else {
                "11.8"
            }
        })
        .map(|v| v.to_string())
}

#[cfg(windows)]
fn windows_ram_total() -> u64 {
    use windows::Win32::System::SystemInformation::MEMORYSTATUSEX;
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe {
        if windows::Win32::System::SystemInformation::GlobalMemoryStatusEx(&mut status).is_ok() {
            (status.ullTotalPhys / 1024 / 1024) as u64
        } else {
            0
        }
    }
}

#[cfg(windows)]
fn windows_ram_used() -> u64 {
    use windows::Win32::System::SystemInformation::MEMORYSTATUSEX;
    let mut status = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    unsafe {
        if windows::Win32::System::SystemInformation::GlobalMemoryStatusEx(&mut status).is_ok() {
            ((status.ullTotalPhys - status.ullAvailPhys) / 1024 / 1024) as u64
        } else {
            0
        }
    }
}

#[cfg(windows)]
fn windows_disk_free() -> u64 {
    use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
    let home = dirs::home_dir().unwrap_or_else(|| std::env::current_dir().unwrap());
    let path = home.as_path().to_str().unwrap_or("C:\\\\");
    let wide: Vec<u16> = path.encode_utf16().chain(Some(0)).collect();
    unsafe {
        let mut free_bytes: u64 = 0;
        let mut total_bytes: u64 = 0;
        let mut _free_total: u64 = 0;
        let free_ptr = &mut free_bytes as *mut u64;
        let total_ptr = &mut total_bytes as *mut u64;
        let total_free_ptr = &mut _free_total as *mut u64;
        if GetDiskFreeSpaceExW(
            windows::core::PCWSTR::from_raw(wide.as_ptr()),
            Some(free_ptr),
            Some(total_ptr),
            Some(total_free_ptr),
        )
        .is_ok()
        {
            free_bytes / 1024 / 1024
        } else {
            0
        }
    }
}

#[cfg(windows)]
fn nvidia_fallback_info() -> Option<(String, u64)> {
    // Fallback for when nvml-wrapper is unavailable
    // Read from Windows registry or WMI
    use std::process::Command;
    let output = Command::new("wmic")
        .args(&[
            "path",
            "Win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion",
        ])
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    for line in stdout.lines() {
        if line.contains("NVIDIA") {
            let name = line.trim().to_string();
            let vram = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
                .map(|v| {
                    if v > 1_000_000 {
                        v / 1024 / 1024
                    } else {
                        v / 1024
                    }
                });
            return Some((name, vram.unwrap_or(4096)));
        }
    }
    None
}

#[cfg(windows)]
fn detect_amd_windows() -> Vec<GpuDevice> {
    use std::process::Command;
    let output = Command::new("wmic")
        .args(&[
            "path",
            "Win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion",
        ])
        .output()
        .ok();
    let stdout = output
        .map(|o| String::from_utf8_lossy(&o.stdout))
        .unwrap_or_default();
    let mut out = Vec::new();
    for line in stdout.lines() {
        if line.contains("AMD") || line.contains("Radeon") {
            let name = line.trim().to_string();
            let vram = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
                .map(|v| {
                    if v > 1_000_000 {
                        v / 1024 / 1024
                    } else {
                        v / 1024
                    }
                });
            out.push(GpuDevice {
                vendor: GpuVendor::Amd,
                name,
                vram_total_mb: vram,
                vram_used_mb: None,
                driver: Some("AMD".to_string()),
                cuda_version: None,
            });
        }
    }
    out
}

#[cfg(windows)]
fn detect_intel_windows() -> Vec<GpuDevice> {
    use std::process::Command;
    let output = Command::new("wmic")
        .args(&[
            "path",
            "Win32_VideoController",
            "get",
            "Name,AdapterRAM,DriverVersion",
        ])
        .output()
        .ok();
    let stdout = output
        .map(|o| String::from_utf8_lossy(&o.stdout))
        .unwrap_or_default();
    let mut out = Vec::new();
    for line in stdout.lines() {
        if line.contains("Intel")
            && (line.contains("Graphics") || line.contains("HD") || line.contains("Iris"))
        {
            let name = line.trim().to_string();
            let vram = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse::<u64>().ok())
                .map(|v| {
                    if v > 1_000_000 {
                        v / 1024 / 1024
                    } else {
                        v / 1024
                    }
                });
            out.push(GpuDevice {
                vendor: GpuVendor::Intel,
                name,
                vram_total_mb: vram.or(Some(2048)), // Integrated GPUs share system RAM
                vram_used_mb: None,
                driver: Some("Intel".to_string()),
                cuda_version: None,
            });
        }
    }
    out
}

// ── Linux ───────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn detect_nvidia_linux() -> Vec<GpuDevice> {
    let mut out = Vec::new();
    if let Ok(nvml) = nvml_wrapper::Nvml::init() {
        if let Ok(count) = nvml.device_count() {
            for i in 0..count {
                if let Ok(dev) = nvml.device_by_index(i) {
                    out.push(GpuDevice {
                        vendor: GpuVendor::Nvidia,
                        name: dev.name().unwrap_or_else(|_| "NVIDIA GPU".to_string()),
                        vram_total_mb: dev.memory_info().ok().map(|m| m.total / 1024 / 1024),
                        vram_used_mb: dev.memory_info().ok().map(|m| m.used / 1024 / 1024),
                        driver: dev.driver_version().ok(),
                        cuda_version: Some(cuda_linux_version()),
                    });
                }
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn cuda_linux_version() -> String {
    // Check nvcc or nvidia-smi for CUDA version
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(&["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
    {
        if let Ok(ver) = String::from_utf8(output.stdout) {
            let v = ver
                .trim()
                .split('.')
                .next()
                .unwrap_or("0")
                .parse::<u32>()
                .unwrap_or(0);
            if v >= 550 {
                "12.8"
            } else if v >= 545 {
                "12.6"
            } else if v >= 535 {
                "12.2"
            } else if v >= 530 {
                "12.1"
            } else if v >= 525 {
                "12.0"
            } else if v >= 520 {
                "11.8"
            } else {
                "11.7"
            }
            .to_string()
        } else {
            "12.0".to_string()
        }
    } else {
        "12.0".to_string()
    }
}

#[cfg(target_os = "linux")]
fn detect_amd_linux() -> Vec<GpuDevice> {
    let mut out = Vec::new();
    // Check for ROCm / AMD GPU via /sys
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        for entry in entries.flatten() {
            if let Ok(name) = std::fs::read_to_string(
                entry.path().join("device/driver/module/drivers/pci:amdgpu"),
            ) {
                if !name.is_empty() {
                    out.push(GpuDevice {
                        vendor: GpuVendor::Amd,
                        name: "AMD Radeon GPU".to_string(),
                        vram_total_mb: Some(12288),
                        vram_used_mb: None,
                        driver: Some("amdgpu".to_string()),
                        cuda_version: Some("ROCm".to_string()),
                    });
                    break;
                }
            }
        }
    }
    // Fallback: check rocm-smi
    if out.is_empty() {
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .args(&["--showmeminfo", "vram"])
            .output()
        {
            if output.status.success() {
                out.push(GpuDevice {
                    vendor: GpuVendor::Amd,
                    name: "AMD Radeon GPU".to_string(),
                    vram_total_mb: Some(12288),
                    vram_used_mb: Some(2048),
                    driver: Some("ROCm".to_string()),
                    cuda_version: Some("ROCm".to_string()),
                });
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn detect_intel_linux() -> Vec<GpuDevice> {
    let mut out = Vec::new();
    if let Ok(output) = std::process::Command::new("lspci").args(&["-nn"]).output() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if line.contains("VGA") && line.contains("Intel") {
                out.push(GpuDevice {
                    vendor: GpuVendor::Intel,
                    name: "Intel Arc/HD Graphics".to_string(),
                    vram_total_mb: Some(4096),
                    vram_used_mb: None,
                    driver: Some("i915".to_string()),
                    cuda_version: Some("XPU".to_string()),
                });
                break;
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn linux_ram_total() -> u64 {
    // Read from /proc/meminfo
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemTotal:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                return kb / 1024;
            }
        }
    }
    0
}

#[cfg(target_os = "linux")]
fn linux_ram_used() -> u64 {
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemAvailable:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                let total = linux_ram_total();
                return if total > 0 { total - kb / 1024 } else { 0 };
            }
        }
    }
    0
}

#[cfg(target_os = "linux")]
fn linux_disk_free() -> u64 {
    // Read from /proc/meminfo
    if let Ok(content) = std::fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if line.starts_with("MemAvailable:") {
                let kb: u64 = line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0);
                return kb / 1024;
            }
        }
    }
    0
}

#[cfg(target_os = "linux")]
fn linux_disk_free() -> u64 {
    if let Ok(output) = std::process::Command::new("df").args(&["-B1", "--output=avail", "/"]) {
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().skip(1) {
            if let Ok(bytes) = line.trim().parse::<u64>() {
                return bytes;
            }
        }
    }
    0
}

// ── Common ──────────────────────────────────────────────────────

fn detect_cpu() -> String {
    #[cfg(windows)]
    {
        // Try to get CPU info from wmic
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["cpu", "get", "name"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                let line = line.trim();
                if !line.is_empty() && !line.starts_with("Name") {
                    return line.to_string();
                }
            }
        }
        "Unknown CPU".to_string()
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in content.lines() {
                if line.starts_with("model name") {
                    return line
                        .split(':')
                        .nth(1)
                        .unwrap_or("Unknown CPU")
                        .trim()
                        .to_string();
                }
            }
        }
        "Unknown CPU".to_string()
    }

    #[cfg(not(any(windows, target_os = "linux")))]
    "Unknown CPU".to_string()
}
