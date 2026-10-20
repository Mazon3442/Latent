/// Frame cache: stores generated frames, thumbnails, and the frame index.
/// Frames live in %LOCALAPPDATA%\com.mazon.latent\cache (Windows) or ~/.cache/latent (Linux).
/// The cache is cleared on app close (Mimic's rule).
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Imported source image (photo, screenshot, frame used as source).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub id: String,
    pub path: PathBuf,
    pub thumb: PathBuf,
    pub width: usize,
    pub height: usize,
    pub name: String,
    pub from_frame: Option<String>,
}

impl Source {
    /// Copy an image from the given path into the cache, producing a thumbnail.
    pub fn import_source(from: &PathBuf, thumb_path: &PathBuf) -> Result<PathBuf, String> {
        let cache_dir = cache_dir();
        let saved = cache_dir.join(
            from.file_name()
                .ok_or("Invalid filename")?
                .to_str()
                .ok_or("Non-UTF8 filename")?,
        );

        // Copy the file
        std::fs::copy(from, &saved).map_err(|e| e.to_string())?;

        // Create thumbnail
        if let Ok(img) = image::open(from) {
            let thumb = image::DynamicImage::ImageRgba8(image::imageops::thumbnail(&img, 100, 100));
            std::fs::create_dir_all(thumb_path.parent().ok_or("No parent")?).ok();
            thumb.save(thumb_path).ok();
        }

        Ok(saved)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    pub id: String,
    pub batch: String,
    pub index: usize,
    pub path: PathBuf,
    pub thumb: PathBuf,
    pub width: usize,
    pub height: usize,
    pub seed: i64,
    pub recipe: GenRequest,
    pub model: String,
    pub variant: String,
    pub elapsed: f64,
    pub saved: Option<String>, // None = unsaved
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenRequest {
    pub model: String,
    pub variant: Option<String>,
    pub prompt: String,
    pub negative: Option<String>,
    pub sources: Vec<String>,
    pub mask: Option<String>,
    pub width: usize,
    pub height: usize,
    pub count: usize,
    pub seed: Option<i64>,
    pub options: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameIndex {
    pub frames: Vec<Frame>,
    pub saved_only: bool, // If true, only persisted frames are returned
}

impl Frame {
    pub fn new(
        id: String,
        batch: String,
        index: usize,
        path: PathBuf,
        width: usize,
        height: usize,
        seed: i64,
        recipe: GenRequest,
        model: String,
        variant: String,
        elapsed: f64,
    ) -> Self {
        let cache_dir = cache_dir();
        let thumb = cache_dir.join(format!("{}.thumb.png", id));
        let created_at = chrono::Utc::now();

        Frame {
            id,
            batch,
            index,
            path,
            thumb,
            width,
            height,
            seed,
            recipe,
            model,
            variant,
            elapsed,
            saved: None,
            created_at,
        }
    }
}

/// Return the cache directory path.
pub fn cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("com.mazon.latent")
            .join("cache")
    }
    #[cfg(not(target_os = "windows"))]
    {
        dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("latent")
    }
}

/// Return the frame index file path.
fn index_path() -> PathBuf {
    dirs::state_dir()
        .unwrap_or_else(|| dirs::config_dir().unwrap_or_else(|| PathBuf::from(".")))
        .join("com.mazon.latent")
        .join("frames.json")
}

/// Load the frame index from disk.
pub fn load_index() -> FrameIndex {
    let path = index_path();
    if path.exists() {
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(index) = serde_json::from_str(&content) {
                return index;
            }
        }
    }
    FrameIndex {
        frames: Vec::new(),
        saved_only: false,
    }
}

/// Save the frame index to disk.
pub fn save_index(index: &FrameIndex) -> Result<(), String> {
    let path = index_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create index dir: {}", e))?;
    }
    let content = serde_json::to_string_pretty(index)
        .map_err(|e| format!("Failed to serialize index: {}", e))?;
    std::fs::write(&path, content).map_err(|e| format!("Failed to write index: {}", e))?;
    Ok(())
}

/// Add a frame to the index and save.
pub fn add_frame(index: &mut FrameIndex, frame: Frame) -> Result<(), String> {
    index.frames.push(frame);
    save_index(index)
}

/// Remove a frame from the index and save.
pub fn remove_frame(index: &mut FrameIndex, frame_id: &str) -> Result<(), String> {
    index.frames.retain(|f| f.id != frame_id);
    save_index(index)
}

/// Get all frames.
pub fn get_frames(index: &FrameIndex) -> Vec<&Frame> {
    if index.saved_only {
        index.frames.iter().filter(|f| f.saved.is_some()).collect()
    } else {
        index.frames.iter().collect()
    }
}

/// Get frames grouped by batch.
pub fn get_batches(index: &FrameIndex) -> Vec<(&str, Vec<&Frame>)> {
    let mut batch_map: HashMap<String, Vec<&Frame>> = HashMap::new();
    for frame in get_frames(index) {
        batch_map
            .entry(frame.batch.clone())
            .or_default()
            .push(frame);
    }
    let mut batches: Vec<_> = batch_map.into_iter().collect();
    // Sort by latest frame in each batch (newest first)
    batches.sort_by(|a, b| {
        let a_latest =
            a.1.iter()
                .map(|f| f.created_at)
                .max()
                .unwrap_or(chrono::Utc::now());
        let b_latest =
            b.1.iter()
                .map(|f| f.created_at)
                .max()
                .unwrap_or(chrono::Utc::now());
        b_latest.cmp(&a_latest)
    });
    batches.into_iter().map(|(k, v)| (k.as_str(), v)).collect()
}

/// Clear all unsaved frames.
pub fn clear_unsaved(index: &mut FrameIndex) {
    index.frames.retain(|f| f.saved.is_some());
    save_index(index).ok();
}

/// Count unsaved frames.
pub fn unsaved_count(index: &FrameIndex) -> usize {
    index.frames.iter().filter(|f| f.saved.is_none()).count()
}
