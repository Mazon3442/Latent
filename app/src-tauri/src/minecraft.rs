//! Saving a frame as a Minecraft GUI texture: the PNG in the resource-pack folder layout, plus the `.png.mcmeta` that
//! tells the game how to scale it.
//!
//! Java 1.20.2 and later draw GUI sprites (`textures/gui/sprites/`) through `blitSprite`, which reads a `gui.scaling`
//! block from the sprite's mcmeta: `stretch` (the default, no file needed), `tile`, or `nine_slice` with a border.
//! Older style screens use a plain texture under `textures/gui/` and need no mcmeta.

use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Export {
    /// The mod's namespace: lowercase letters, digits, `_`, `-`, `.`.
    pub modid: String,
    /// Path under the namespace, like `widget/button` or `container/altar`: lowercase, digits, `_`, `-`, `.`, `/`.
    pub name: String,
    /// `sprite` (textures/gui/sprites, scaled by mcmeta) or `texture` (textures/gui, drawn at its own size).
    pub kind: String,
    /// For sprites: `stretch`, `tile` or `nine_slice`.
    pub scaling: String,
    /// Nine-slice border widths in pixels: left, top, right, bottom.
    pub border: [u32; 4],
}

impl Default for Export {
    fn default() -> Self {
        Export { modid: String::new(), name: String::new(), kind: "sprite".into(), scaling: "stretch".into(), border: [4, 4, 4, 4] }
    }
}

fn valid_part(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '_' | '-' | '.'))
}

fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.split('/').all(|p| valid_part(p) && p != "." && p != "..") && !s.starts_with('/') && !s.ends_with('/')
}

impl Export {
    pub fn check(&self, w: u32, h: u32) -> Result<(), String> {
        if !valid_part(&self.modid) || self.modid.contains('.') {
            return Err("The mod id can use lowercase letters, digits, _ and - only.".into());
        }
        if !valid_name(&self.name) {
            return Err("The texture name can use lowercase letters, digits, _ - . and / between folders.".into());
        }
        if !matches!(self.kind.as_str(), "sprite" | "texture") {
            return Err("Kind must be sprite or texture.".into());
        }
        if self.kind == "sprite" {
            match self.scaling.as_str() {
                "stretch" | "tile" => {}
                "nine_slice" => {
                    let [l, t, r, b] = self.border;
                    if l + r >= w || t + b >= h {
                        return Err(format!("The border leaves nothing to stretch in a {w} × {h} sprite."));
                    }
                }
                _ => return Err("Scaling must be stretch, tile or nine_slice.".into()),
            }
        }
        Ok(())
    }

    /// Where the PNG goes under `root` (the Minecraft export folder).
    pub fn png_path(&self, root: &Path) -> PathBuf {
        let mut p = root.join("assets").join(&self.modid).join("textures").join("gui");
        if self.kind == "sprite" {
            p.push("sprites");
        }
        for part in self.name.split('/') {
            p.push(part);
        }
        p.set_extension("png");
        p
    }

    /// The mcmeta for a sprite of this size, if it needs one.
    pub fn mcmeta(&self, w: u32, h: u32) -> Option<String> {
        if self.kind != "sprite" {
            return None;
        }
        let scaling = match self.scaling.as_str() {
            "tile" => serde_json::json!({ "type": "tile", "width": w, "height": h }),
            "nine_slice" => {
                let [left, top, right, bottom] = self.border;
                serde_json::json!({ "type": "nine_slice", "width": w, "height": h,
                    "border": { "left": left, "top": top, "right": right, "bottom": bottom } })
            }
            _ => return None,
        };
        serde_json::to_string_pretty(&serde_json::json!({ "gui": { "scaling": scaling } })).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nine() -> Export {
        Export { modid: "bodaddon".into(), name: "widget/panel".into(), scaling: "nine_slice".into(), ..Export::default() }
    }

    #[test]
    fn sprite_lands_under_gui_sprites_with_a_nine_slice_mcmeta() {
        let e = nine();
        e.check(32, 32).unwrap();
        assert_eq!(e.png_path(Path::new("out")), Path::new("out/assets/bodaddon/textures/gui/sprites/widget/panel.png"));
        let meta: serde_json::Value = serde_json::from_str(&e.mcmeta(32, 24).unwrap()).unwrap();
        let s = &meta["gui"]["scaling"];
        assert_eq!((s["type"].as_str(), s["width"].as_u64(), s["height"].as_u64()), (Some("nine_slice"), Some(32), Some(24)));
        assert_eq!(s["border"]["right"], 4);
    }

    #[test]
    fn texture_has_no_mcmeta_and_sits_beside_the_sprites_folder() {
        let e = Export { modid: "m".into(), name: "container/altar".into(), kind: "texture".into(), ..Export::default() };
        e.check(256, 256).unwrap();
        assert_eq!(e.png_path(Path::new("out")), Path::new("out/assets/m/textures/gui/container/altar.png"));
        assert!(e.mcmeta(256, 256).is_none());
    }

    #[test]
    fn stretch_needs_no_file_and_tile_records_the_size() {
        let mut e = nine();
        e.scaling = "stretch".into();
        assert!(e.mcmeta(16, 16).is_none());
        e.scaling = "tile".into();
        assert!(e.mcmeta(16, 16).unwrap().contains("\"tile\""));
    }

    #[test]
    fn rejects_names_that_could_leave_the_folder_or_break_the_game() {
        for bad in ["../x", "a//b", "/a", "a/", "A", "a b", "a/./b", "a\\b", ""] {
            let e = Export { name: bad.into(), ..nine() };
            assert!(e.check(32, 32).is_err(), "{bad:?} should be refused");
        }
        for bad in ["Mod", "my mod", "a.b", "../a", ""] {
            let e = Export { modid: bad.into(), ..nine() };
            assert!(e.check(32, 32).is_err(), "{bad:?} should be refused as a mod id");
        }
    }

    #[test]
    fn a_border_must_leave_a_middle() {
        let e = Export { border: [16, 4, 16, 4], ..nine() };
        assert!(e.check(32, 32).is_err());
        assert!(e.check(40, 32).is_ok());
    }
}
