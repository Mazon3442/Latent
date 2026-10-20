//! Sources (photos brought in to edit) and frames (what came out).
//!
//! Sources are copied into the app data folder, so moving the original never breaks a recipe. Frames live in a cache
//! folder that is emptied when Latent starts and closes: nothing is kept unless it's saved.

use crate::store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

/// Sources larger than this are scaled down on import; no model here works above it.
const MAX_SIDE: u32 = 2048;

#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct Source {
    pub id: String,
    pub path: String,
    pub width: u32,
    pub height: u32,
    pub name: String,
    pub from_frame: Option<String>,
}

#[derive(Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct Frame {
    pub id: String,
    pub batch: String,
    pub index: u32,
    pub path: String,
    pub thumb: String,
    pub width: u32,
    pub height: u32,
    pub seed: i64,
    pub recipe: Value,
    pub model: String,
    pub variant: String,
    pub elapsed: f64,
    pub saved: Option<String>,
    pub created_at: u128,
}

pub struct Library {
    sources_dir: PathBuf,
    sources_file: PathBuf,
    pub frames_dir: PathBuf,
    sources: Mutex<Vec<Source>>,
    frames: Mutex<Vec<Frame>>,
}

pub fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

impl Library {
    pub fn open(data: &Path, frames_dir: PathBuf) -> Result<Library, String> {
        let sources_dir = data.join("sources");
        let _ = std::fs::remove_dir_all(&frames_dir); // frames from last session were never saved
        for d in [&sources_dir, &frames_dir] {
            std::fs::create_dir_all(d).map_err(|e| format!("can't create {}: {e}", d.display()))?;
        }
        let sources_file = data.join("sources.json");
        let sources: Vec<Source> = store::load(&sources_file);
        let sources = sources.into_iter().filter(|s| Path::new(&s.path).is_file()).collect();
        Ok(Library { sources_dir, sources_file, frames_dir, sources: Mutex::new(sources), frames: Mutex::new(Vec::new()) })
    }

    pub fn sources(&self) -> Vec<Source> {
        self.sources.lock().unwrap().clone()
    }

    pub fn frames(&self) -> Vec<Frame> {
        self.frames.lock().unwrap().clone()
    }

    pub fn source(&self, id: &str) -> Result<Source, String> {
        self.sources.lock().unwrap().iter().find(|s| s.id == id).cloned().ok_or_else(|| "That source is gone".into())
    }

    pub fn frame(&self, id: &str) -> Result<Frame, String> {
        self.frames.lock().unwrap().iter().find(|f| f.id == id).cloned().ok_or_else(|| "That frame is gone".into())
    }

    /// Decodes, scales down if needed, and stores a copy as PNG.
    fn add_source(&self, image: image::DynamicImage, name: &str, from_frame: Option<String>) -> Result<Source, String> {
        let image = if image.width().max(image.height()) > MAX_SIDE {
            image.resize(MAX_SIDE, MAX_SIDE, image::imageops::FilterType::Lanczos3)
        } else {
            image
        };
        let id = format!("s{}", now_ms());
        let path = self.sources_dir.join(format!("{id}.png"));
        image.to_rgb8().save(&path).map_err(|e| format!("can't store the source: {e}"))?;
        let source = Source {
            id,
            path: path.to_string_lossy().to_string(),
            width: image.width(),
            height: image.height(),
            name: name.to_string(),
            from_frame,
        };
        let mut all = self.sources.lock().unwrap();
        all.insert(0, source.clone());
        store::save(&self.sources_file, &*all)?;
        Ok(source)
    }

    pub fn import(&self, path: &Path) -> Result<Source, String> {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let image = image::open(path).map_err(|e| format!("{name} isn't an image Latent can read ({e})"))?;
        self.add_source(image, &name, None)
    }

    pub fn paste(&self, bytes: &[u8]) -> Result<Source, String> {
        let image = image::load_from_memory(bytes).map_err(|e| format!("the clipboard image can't be read ({e})"))?;
        self.add_source(image, "Pasted image", None)
    }

    pub fn source_from_frame(&self, id: &str) -> Result<Source, String> {
        let frame = self.frame(id)?;
        let image = image::open(&frame.path).map_err(|e| e.to_string())?;
        let name = format!("Frame {} · seed {}", frame.index + 1, frame.seed);
        self.add_source(image, &name, Some(frame.id))
    }

    pub fn delete_source(&self, id: &str) -> Result<(), String> {
        let mut all = self.sources.lock().unwrap();
        if let Some(i) = all.iter().position(|s| s.id == id) {
            let _ = std::fs::remove_file(&all[i].path);
            all.remove(i);
        }
        store::save(&self.sources_file, &*all)
    }

    pub fn add_frame(&self, frame: Frame) {
        self.frames.lock().unwrap().push(frame);
    }

    pub fn mark_saved(&self, id: &str, path: &str) {
        if let Some(f) = self.frames.lock().unwrap().iter_mut().find(|f| f.id == id) {
            f.saved = Some(path.to_string());
        }
    }

    pub fn delete_frame(&self, id: &str) {
        let mut all = self.frames.lock().unwrap();
        if let Some(i) = all.iter().position(|f| f.id == id) {
            let f = all.remove(i);
            for p in [f.path.clone(), f.thumb, format!("{}.p.jpg", f.path)] {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    pub fn clear_frames(&self) {
        self.frames.lock().unwrap().clear();
        let _ = std::fs::remove_dir_all(&self.frames_dir);
    }
}
