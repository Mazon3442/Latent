//! The pixel tool: turns any image into clean, game-ready pixel art.
//!
//! Diffusion models draw "pixel style" images whose pixels are blurry, off-grid and colored by thousands of near
//! duplicates. Three steps make them real pixel art, and they are plain image processing (no model, no GPU):
//!
//! 1. **Grid.** The image is cut into one cell per output pixel, and each cell becomes the color most of it is made of
//!    (a two-means clustering per cell, the "k-centroid" downscale). That keeps edges crisp where averaging smears them.
//! 2. **Palette.** Colors snap to a small palette: one you give (hex lines or a GIMP .gpl) or one found in the image
//!    by median cut.
//! 3. **Cleanup.** Lone pixels are absorbed by their neighbors and, if asked, sprites get a dark outline.
//!
//! Minecraft textures are 1:1, so the result is saved at its true size; the UI shows it with hard edges.

use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Options {
    /// Pixels along the longer side of the result, 8 to 512. 0 finds the picture's own pixel size (64 if it has none).
    pub size: u32,
    /// Palette size when none is given, 2 to 64.
    pub colors: u32,
    /// A palette as text: one #rrggbb per line, or a GIMP .gpl. Replaces `colors` when it has at least two colors.
    pub palette: String,
    /// Absorb single pixels that differ from all four neighbors.
    pub clean: bool,
    /// Dark outline on transparent pixels that touch the sprite.
    pub outline: bool,
    /// `keep` the picture's proportions, or make a square by cutting the middle (`crop`) or adding see-through margins (`pad`).
    pub fit: String,
}

impl Default for Options {
    fn default() -> Self {
        Options { size: 64, colors: 16, palette: String::new(), clean: true, outline: false, fit: "keep".into() }
    }
}

impl Options {
    pub fn check(&self) -> Result<(), String> {
        if self.size != 0 && !(8..=512).contains(&self.size) {
            return Err("Size must be Auto or between 8 and 512 pixels.".into());
        }
        if !matches!(self.fit.as_str(), "keep" | "crop" | "pad") {
            return Err("Shape must be keep, crop or pad.".into());
        }
        if !(2..=64).contains(&self.colors) {
            return Err("Colors must be between 2 and 64.".into());
        }
        if self.palette.len() > 64 * 1024 {
            return Err("That palette is too long.".into());
        }
        Ok(())
    }
}

type Rgb = [u8; 3];

/// Colors from text: `#rrggbb` / `rrggbb` anywhere on a line (Lospec .hex, .txt), or the `R G B name` rows of a .gpl.
pub fn parse_palette(text: &str) -> Vec<Rgb> {
    let mut out: Vec<Rgb> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("GIMP Palette") || line.starts_with("Name:") || line.starts_with("Columns:") {
            continue;
        }
        let hex = line.trim_start_matches('#');
        let c = if hex.len() >= 6 && hex.is_char_boundary(6) && hex[..6].chars().all(|c| c.is_ascii_hexdigit()) {
            let v = u32::from_str_radix(&hex[..6], 16).unwrap_or(0);
            Some([(v >> 16) as u8, (v >> 8) as u8, v as u8])
        } else {
            let mut n = line.split_whitespace().take(3).map(|t| t.parse::<u8>());
            match (n.next(), n.next(), n.next()) {
                (Some(Ok(r)), Some(Ok(g)), Some(Ok(b))) => Some([r, g, b]),
                _ => None,
            }
        };
        if let Some(c) = c {
            if !out.contains(&c) {
                out.push(c);
            }
        }
        if out.len() >= 256 {
            break;
        }
    }
    out
}

fn dist(a: Rgb, b: Rgb) -> i32 {
    // Weighted so green counts most and blue least, closer to how colors look than plain RGB distance.
    let (dr, dg, db) = (a[0] as i32 - b[0] as i32, a[1] as i32 - b[1] as i32, a[2] as i32 - b[2] as i32);
    2 * dr * dr + 4 * dg * dg + 3 * db * db
}

fn luma(c: Rgb) -> i32 {
    299 * c[0] as i32 + 587 * c[1] as i32 + 114 * c[2] as i32
}

/// The color most of `cell` is made of: two-means on the pixels, then the average of the bigger group.
fn dominant(cell: &[Rgb]) -> Rgb {
    let lo = *cell.iter().min_by_key(|c| luma(**c)).unwrap();
    let hi = *cell.iter().max_by_key(|c| luma(**c)).unwrap();
    if lo == hi {
        return lo;
    }
    let (mut a, mut b) = (lo, hi);
    let mut group = vec![false; cell.len()];
    for _ in 0..4 {
        let (mut sa, mut sb, mut na, mut nb) = ([0u32; 3], [0u32; 3], 0u32, 0u32);
        for (i, c) in cell.iter().enumerate() {
            group[i] = dist(*c, b) < dist(*c, a);
            let (s, n) = if group[i] { (&mut sb, &mut nb) } else { (&mut sa, &mut na) };
            for k in 0..3 {
                s[k] += c[k] as u32;
            }
            *n += 1;
        }
        if na > 0 {
            a = [(sa[0] / na) as u8, (sa[1] / na) as u8, (sa[2] / na) as u8];
        }
        if nb > 0 {
            b = [(sb[0] / nb) as u8, (sb[1] / nb) as u8, (sb[2] / nb) as u8];
        }
    }
    let nb = group.iter().filter(|g| **g).count();
    if nb * 2 > cell.len() { b } else { a }
}

/// Step 1: the image as `w` x `h` pixels, each the dominant color of its cell. Cells that are mostly see-through stay so.
fn downscale(src: &RgbaImage, w: u32, h: u32) -> Vec<Option<Rgb>> {
    let (sw, sh) = src.dimensions();
    let mut out = Vec::with_capacity((w * h) as usize);
    let mut cell: Vec<Rgb> = Vec::new();
    for y in 0..h {
        let (y0, y1) = (y * sh / h, ((y + 1) * sh / h).max(y * sh / h + 1).min(sh));
        for x in 0..w {
            let (x0, x1) = (x * sw / w, ((x + 1) * sw / w).max(x * sw / w + 1).min(sw));
            cell.clear();
            let mut see_through = 0usize;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let p = src.get_pixel(sx, sy).0;
                    if p[3] < 128 {
                        see_through += 1;
                    } else {
                        cell.push([p[0], p[1], p[2]]);
                    }
                }
            }
            let total = ((y1 - y0) * (x1 - x0)) as usize;
            out.push(if cell.is_empty() || see_through * 2 > total { None } else { Some(dominant(&cell)) });
        }
    }
    out
}

/// Step 2a: `n` colors that stand for `pixels`, by median cut (split the widest box at its median until there are n).
fn median_cut(pixels: &[Rgb], n: usize) -> Vec<Rgb> {
    if pixels.is_empty() {
        return Vec::new();
    }
    let mut boxes: Vec<Vec<Rgb>> = vec![pixels.to_vec()];
    while boxes.len() < n {
        // The box with the widest spread in any channel, if it can still be split.
        let range = |b: &Vec<Rgb>| (0..3).map(|k| {
            let (lo, hi) = b.iter().fold((255u8, 0u8), |(lo, hi), c| (lo.min(c[k]), hi.max(c[k])));
            (hi - lo, k)
        }).max().unwrap();
        let Some((i, (spread, k))) = boxes.iter().enumerate().map(|(i, b)| (i, range(b))).max_by_key(|(_, (s, _))| *s) else { break };
        if spread == 0 || boxes[i].len() < 2 {
            break;
        }
        let mut b = boxes.swap_remove(i);
        b.sort_by_key(|c| c[k]);
        let upper = b.split_off(b.len() / 2);
        boxes.push(b);
        boxes.push(upper);
    }
    let mut out: Vec<Rgb> = boxes
        .iter()
        .map(|b| {
            let mut s = [0u64; 3];
            for c in b {
                for k in 0..3 {
                    s[k] += c[k] as u64;
                }
            }
            let n = b.len() as u64;
            [(s[0] / n) as u8, (s[1] / n) as u8, (s[2] / n) as u8]
        })
        .collect();
    out.sort_by_key(|c| luma(*c));
    out.dedup();
    out
}

fn nearest(palette: &[Rgb], c: Rgb) -> Rgb {
    *palette.iter().min_by_key(|p| dist(**p, c)).unwrap_or(&c)
}

/// Step 3a: a pixel that none of its four neighbors share, whose neighbors all agree, takes their color.
fn despeckle(px: &mut [Option<Rgb>], w: usize, h: usize) {
    let copy = px.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let n = [copy[(y - 1) * w + x], copy[(y + 1) * w + x], copy[y * w + x - 1], copy[y * w + x + 1]];
            if n[0] == n[1] && n[1] == n[2] && n[2] == n[3] && copy[y * w + x] != n[0] {
                px[y * w + x] = n[0];
            }
        }
    }
}

/// Step 3b: transparent pixels next to the sprite become a darkened copy of the darkest palette color.
fn outline(px: &mut [Option<Rgb>], w: usize, h: usize, palette: &[Rgb]) {
    let ink = palette.iter().min_by_key(|c| luma(**c)).map(|c| [c[0] / 2, c[1] / 2, c[2] / 2]).unwrap_or([0, 0, 0]);
    let copy = px.to_vec();
    for y in 0..h {
        for x in 0..w {
            if copy[y * w + x].is_some() {
                continue;
            }
            let touches = (x > 0 && copy[y * w + x - 1].is_some())
                || (x + 1 < w && copy[y * w + x + 1].is_some())
                || (y > 0 && copy[(y - 1) * w + x].is_some())
                || (y + 1 < h && copy[(y + 1) * w + x].is_some());
            if touches {
                px[y * w + x] = Some(ink);
            }
        }
    }
}

/// How far ink touching a rectangle is followed beyond it.
const SNAP: usize = 3;

/// Erases rectangles (x, y, w, h) from finished pixel art: each is filled by smoothly blending the pixels around it, then
/// snapped back to the colors the image already uses, so no new colors appear. For taking text, swatches and arrows off
/// a generated screen so the game can draw its own.
///
/// A rectangle doesn't have to be exact. Pixels darker than the paper around it (text, outlines, shadows) that touch it
/// are taken too, up to a few pixels beyond its edge, so a loose box leaves no half-erased letters to smear into the fill.
/// Nothing else outside the rectangles changes.
pub fn erase(img: &RgbaImage, rects: &[[u32; 4]]) -> Result<RgbaImage, String> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if rects.is_empty() || rects.len() > 500 {
        return Err("Mark between 1 and 500 areas to erase.".into());
    }
    let mut palette: Vec<Rgb> = Vec::new();
    for p in img.pixels().filter(|p| p[3] >= 128) {
        let c = [p[0], p[1], p[2]];
        if !palette.contains(&c) {
            palette.push(c);
            if palette.len() > 256 {
                return Err("This image has too many colors to fill from. Run it through the pixel tool first.".into());
            }
        }
    }
    // 0 not part of the picture (see-through), 1 known, 2 to fill.
    let mut state: Vec<u8> = img.pixels().map(|p| if p[3] >= 128 { 1 } else { 0 }).collect();
    let mut value: Vec<[f32; 3]> = img.pixels().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]).collect();
    let mut any = false;
    for &[rx, ry, rw, rh] in rects {
        for y in ry as usize..(ry as usize).saturating_add(rh as usize).min(h) {
            for x in rx as usize..(rx as usize).saturating_add(rw as usize).min(w) {
                if state[y * w + x] == 1 {
                    state[y * w + x] = 2;
                    any = true;
                }
            }
        }
    }
    if !any {
        return Err("None of those areas touch the picture.".into());
    }
    let luma = |i: usize| {
        let p = img.get_pixel((i % w) as u32, (i / w) as u32);
        (299 * p[0] as i32 + 587 * p[1] as i32 + 114 * p[2] as i32) / 1000
    };
    for &[rx, ry, rw, rh] in rects {
        let (x0, y0) = (rx as usize, ry as usize);
        let (x1, y1) = (x0.saturating_add(rw as usize).min(w), y0.saturating_add(rh as usize).min(h));
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        // The paper: the middle brightness of what lies just outside the rectangle.
        let (ox0, oy0, ox1, oy1) = (x0.saturating_sub(2), y0.saturating_sub(2), (x1 + 2).min(w), (y1 + 2).min(h));
        let mut ring: Vec<i32> = Vec::new();
        for y in oy0..oy1 {
            for x in ox0..ox1 {
                if (x < x0 || x >= x1 || y < y0 || y >= y1) && state[y * w + x] == 1 {
                    ring.push(luma(y * w + x));
                }
            }
        }
        if ring.is_empty() {
            continue;
        }
        ring.sort_unstable();
        let ink_below = ring[ring.len() / 2] - 28;
        // Grow from the rectangle through darker pixels, a few pixels at most.
        let (lx0, ly0, lx1, ly1) = (x0.saturating_sub(SNAP), y0.saturating_sub(SNAP), (x1 + SNAP).min(w), (y1 + SNAP).min(h));
        let mut queue: Vec<usize> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| y * w + x)).filter(|i| state[*i] == 2).collect();
        while let Some(i) = queue.pop() {
            let (x, y) = (i % w, i / w);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                    if nx < lx0 as i32 || ny < ly0 as i32 || nx >= lx1 as i32 || ny >= ly1 as i32 {
                        continue;
                    }
                    let j = ny as usize * w + nx as usize;
                    if state[j] == 1 && luma(j) < ink_below {
                        state[j] = 2;
                        queue.push(j);
                    }
                }
            }
        }
    }
    let neighbors = |i: usize| {
        let (x, y) = (i % w, i / w);
        [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)]
    };
    // Start each area from the colors just outside it, working inward a ring at a time...
    let mut resolved: Vec<bool> = state.iter().map(|s| *s == 1).collect();
    loop {
        let mut ring = Vec::new();
        for i in 0..w * h {
            if state[i] != 2 || resolved[i] {
                continue;
            }
            let (mut sum, mut n) = ([0f32; 3], 0f32);
            for j in neighbors(i).into_iter().flatten() {
                if resolved[j] {
                    for k in 0..3 {
                        sum[k] += value[j][k];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                ring.push((i, [sum[0] / n, sum[1] / n, sum[2] / n]));
            }
        }
        if ring.is_empty() {
            break;
        }
        for (i, v) in ring {
            value[i] = v;
            resolved[i] = true;
        }
    }
    // ...then let it settle into a smooth blend.
    let fill: Vec<usize> = (0..w * h).filter(|i| state[*i] == 2 && resolved[*i]).collect();
    if fill.is_empty() {
        return Err("There is nothing around those areas to fill from.".into());
    }
    for _ in 0..400 {
        let prev = value.clone();
        for &i in &fill {
            let (mut sum, mut n) = ([0f32; 3], 0f32);
            for j in neighbors(i).into_iter().flatten() {
                if resolved[j] {
                    for k in 0..3 {
                        sum[k] += prev[j][k];
                    }
                    n += 1.0;
                }
            }
            if n > 0.0 {
                value[i] = [sum[0] / n, sum[1] / n, sum[2] / n];
            }
        }
    }
    let mut out = img.clone();
    for &i in &fill {
        let c = nearest(&palette, [value[i][0].round() as u8, value[i][1].round() as u8, value[i][2].round() as u8]);
        out.put_pixel((i % w) as u32, (i / w) as u32, Rgba([c[0], c[1], c[2], 255]));
    }
    Ok(out)
}

/// Writes `img` as PNG with the marks every Latent image carries: the recipe, the software, and (when what went in was
/// itself AI-generated) the AI-generated flag.
pub fn write_png(path: &std::path::Path, img: &RgbaImage, recipe: &serde_json::Value, ai: bool) -> Result<(), String> {
    let fail = |e: String| format!("can't write {}: {e}", path.display());
    let file = std::fs::File::create(path).map_err(|e| fail(e.to_string()))?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), img.width(), img.height());
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.add_text_chunk("Software".into(), "Latent".into()).map_err(|e| fail(e.to_string()))?;
    if ai {
        enc.add_text_chunk("ai_generated".into(), "true".into()).map_err(|e| fail(e.to_string()))?;
    }
    enc.add_itxt_chunk("parameters".into(), recipe.to_string()).map_err(|e| fail(e.to_string()))?;
    let mut w = enc.write_header().map_err(|e| fail(e.to_string()))?;
    w.write_image_data(img.as_raw()).map_err(|e| fail(e.to_string()))
}

// ---------------------------------------------------------------- finding the pixel size

/// How much the picture changes across each column boundary (or row boundary), summed down the image. On pixel art
/// upscaled to a larger canvas the changes pile up on a regular lattice, one line per original pixel.
fn edge_profile(img: &RgbaImage, along_x: bool) -> Vec<f32> {
    let (w, h) = img.dimensions();
    let (len, cross) = if along_x { (w, h) } else { (h, w) };
    let step = (cross / 240).max(1);
    let mut e = vec![0f32; len as usize];
    for t in (0..cross).step_by(step as usize) {
        for i in 1..len {
            let (a, b) = if along_x { (img.get_pixel(i - 1, t), img.get_pixel(i, t)) } else { (img.get_pixel(t, i - 1), img.get_pixel(t, i)) };
            e[i as usize] += (0..3).map(|k| (a[k] as f32 - b[k] as f32).abs()).sum::<f32>();
        }
    }
    e
}

/// How much stronger the edges are on a lattice of spacing `p` (best of its offsets) than on average. About 1 means the
/// lattice means nothing; real pixel art gives several times that.
fn lattice_score(e: &[f32], p: f32) -> f32 {
    let mean = e.iter().sum::<f32>() / e.len() as f32;
    if mean <= f32::EPSILON {
        return 0.0;
    }
    let phases = ((p * 2.0).ceil() as usize).clamp(4, 48);
    let mut best = 0f32;
    for k in 0..phases {
        let start = p * k as f32 / phases as f32;
        let (mut sum, mut n) = (0f32, 0f32);
        let mut x = start;
        while (x.round() as usize) < e.len() {
            sum += e[x.round() as usize];
            n += 1.0;
            x += p;
        }
        if n > 0.0 {
            best = best.max(sum / n);
        }
    }
    best / mean
}

/// The size, in picture pixels, of one pixel of the original art, if the image is pixel art enlarged by a steady
/// factor (it may be a fraction, like 5.4). None when the picture doesn't sit on a grid.
pub fn detect_cell(img: &RgbaImage) -> Option<f32> {
    let (w, h) = img.dimensions();
    if w < 32 || h < 32 {
        return None;
    }
    let (ex, ey) = (edge_profile(img, true), edge_profile(img, false));
    let max_p = (w.min(h) as f32 / 8.0).min(64.0);
    let mut scores: Vec<(f32, f32)> = Vec::new();
    let mut p = 2.0f32;
    while p <= max_p {
        scores.push((p, (lattice_score(&ex, p) + lattice_score(&ey, p)) / 2.0));
        p += 0.02;
    }
    let best = scores.iter().map(|s| s.1).fold(0f32, f32::max);
    if best < 2.0 {
        return None;
    }
    // Multiples of the true spacing score nearly as well, and half of it (a lattice twice as fine) scores about half. So
    // the real spacing is the smallest one that scores well: past a clear majority of the best, short of that half.
    let first = scores.iter().position(|s| s.1 >= best * 0.65)?;
    let window = scores[first..].iter().take_while(|s| s.0 <= scores[first].0 * 1.05);
    window.max_by(|a, b| a.1.total_cmp(&b.1)).map(|s| s.0)
}

/// The long side, in pixels, of the original art, if the picture is an enlargement of pixel art.
pub fn detect_size(img: &RgbaImage) -> Option<u32> {
    detect_cell(img).map(|p| ((img.width().max(img.height()) as f32 / p).round() as u32).clamp(8, 512))
}

/// What size an image already is as pixel art: its detected size when it is an enlargement, or its own size when it is
/// small with few colors (pixel art at 1:1). None for photos and renders.
pub fn native_size(img: &RgbaImage) -> Option<u32> {
    if let Some(n) = detect_size(img) {
        return Some(n);
    }
    let long = img.width().max(img.height());
    if !(8..=512).contains(&long) {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    for p in img.pixels() {
        seen.insert(p.0);
        if seen.len() > 512 {
            return None;
        }
    }
    Some(long)
}

/// What to do to every new frame so edits stay pixel art (the "Keep results pixelated" switch).
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Spec {
    /// Pixels along the longer side; 0 matches the source (or 64 for a new image).
    pub size: u32,
    /// Palette size when the source's colors aren't kept.
    pub colors: u32,
    /// Use the colors of the source picture, so an edit doesn't drift the palette.
    pub keep_palette: bool,
    /// `keep`, `crop` or `pad`, as in `Options`.
    pub fit: String,
    pub clean: bool,
}

impl Default for Spec {
    fn default() -> Self {
        Spec { size: 0, colors: 16, keep_palette: true, fit: "keep".into(), clean: true }
    }
}

impl Spec {
    pub fn check(&self) -> Result<(), String> {
        self.options(None).check()
    }

    /// The pixel options for one frame, given the picture it was made from (if any).
    pub fn options(&self, source: Option<&RgbaImage>) -> Options {
        let size = if self.size > 0 { self.size } else { source.and_then(native_size).unwrap_or(64) };
        let mut palette = String::new();
        if let (true, Some(src)) = (self.keep_palette, source) {
            let colors: Vec<Rgb> = src.pixels().filter(|p| p[3] >= 128).map(|p| [p[0], p[1], p[2]]).collect();
            let mut unique: Vec<Rgb> = Vec::new();
            for c in &colors {
                if !unique.contains(c) {
                    unique.push(*c);
                    if unique.len() > 64 {
                        break;
                    }
                }
            }
            if unique.len() > 64 {
                unique = median_cut(&colors, self.colors.clamp(2, 64) as usize);
            }
            palette = unique.iter().map(|c| format!("#{:02x}{:02x}{:02x}\n", c[0], c[1], c[2])).collect();
        }
        Options { size, colors: self.colors, palette, clean: self.clean, outline: false, fit: self.fit.clone() }
    }
}

/// The picture as a square: its middle cut out (`crop`) or centered on see-through margins (`pad`).
fn squared(src: &RgbaImage, fit: &str) -> RgbaImage {
    let (w, h) = src.dimensions();
    if w == h || fit == "keep" {
        return src.clone();
    }
    if fit == "crop" {
        let side = w.min(h);
        return image::imageops::crop_imm(src, (w - side) / 2, (h - side) / 2, side, side).to_image();
    }
    let side = w.max(h);
    let mut out = RgbaImage::new(side, side);
    image::imageops::replace(&mut out, src, ((side - w) / 2) as i64, ((side - h) / 2) as i64);
    out
}

pub fn run(src: &RgbaImage, o: &Options) -> Result<RgbaImage, String> {
    o.check()?;
    if src.width() == 0 || src.height() == 0 {
        return Err("That image is empty.".into());
    }
    // Auto: the picture's own pixel size, found before it is cropped or padded.
    let size = if o.size == 0 { detect_size(src).unwrap_or(64) } else { o.size };
    let squared_src = squared(src, &o.fit);
    let src = &squared_src;
    let (sw, sh) = src.dimensions();
    let scale = size as f32 / sw.max(sh) as f32;
    let (w, h) = (((sw as f32 * scale).round() as u32).max(1), ((sh as f32 * scale).round() as u32).max(1));
    let mut px = downscale(src, w, h);

    let mut palette = parse_palette(&o.palette);
    if palette.len() < 2 {
        let opaque: Vec<Rgb> = px.iter().flatten().copied().collect();
        palette = median_cut(&opaque, o.colors as usize);
    }
    for p in px.iter_mut().flatten() {
        *p = nearest(&palette, *p);
    }
    if o.clean {
        despeckle(&mut px, w as usize, h as usize);
    }
    if o.outline {
        outline(&mut px, w as usize, h as usize, &palette);
    }

    let mut out = RgbaImage::new(w, h);
    for (i, p) in px.iter().enumerate() {
        let (x, y) = (i as u32 % w, i as u32 / w);
        out.put_pixel(x, y, p.map_or(Rgba([0, 0, 0, 0]), |c| Rgba([c[0], c[1], c[2], 255])));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 64 x 64 image of four flat quadrants with a blurry seam and noise, like a diffusion "pixel" render.
    fn noisy() -> RgbaImage {
        let mut img = RgbaImage::new(64, 64);
        for y in 0..64u32 {
            for x in 0..64u32 {
                let base: Rgb = match (x < 32, y < 32) {
                    (true, true) => [200, 40, 40],
                    (false, true) => [40, 200, 40],
                    (true, false) => [40, 40, 200],
                    (false, false) => [230, 230, 60],
                };
                let n = ((x * 7 + y * 13) % 9) as i32 - 4;
                let c = base.map(|v| (v as i32 + n).clamp(0, 255) as u8);
                img.put_pixel(x, y, Rgba([c[0], c[1], c[2], 255]));
            }
        }
        img
    }

    #[test]
    fn four_quadrants_become_four_flat_colors() {
        let o = Options { size: 8, colors: 4, ..Options::default() };
        let out = run(&noisy(), &o).unwrap();
        assert_eq!(out.dimensions(), (8, 8));
        let colors: std::collections::HashSet<_> = out.pixels().map(|p| p.0).collect();
        assert_eq!(colors.len(), 4, "noise must collapse into the four palette colors");
        assert_eq!(out.get_pixel(0, 0), out.get_pixel(3, 3));
        assert_ne!(out.get_pixel(0, 0), out.get_pixel(7, 0));
    }

    #[test]
    fn keeps_aspect_ratio_and_long_side() {
        let img = RgbaImage::from_pixel(200, 100, Rgba([10, 20, 30, 255]));
        let out = run(&img, &Options { size: 32, ..Options::default() }).unwrap();
        assert_eq!(out.dimensions(), (32, 16));
    }

    #[test]
    fn a_given_palette_is_the_only_source_of_colors() {
        let o = Options { size: 8, palette: "#000000\n#ffffff\n".into(), ..Options::default() };
        let out = run(&noisy(), &o).unwrap();
        assert!(out.pixels().all(|p| p.0 == [0, 0, 0, 255] || p.0 == [255, 255, 255, 255]));
    }

    #[test]
    fn parses_hex_lines_and_gpl() {
        assert_eq!(parse_palette("#ff0000\n00ff00 green\n#ff0000\n"), vec![[255, 0, 0], [0, 255, 0]]);
        assert_eq!(parse_palette("GIMP Palette\nName: x\n#\n 10  20  30 a\n255 255 255 b\n"), vec![[10, 20, 30], [255, 255, 255]]);
    }

    #[test]
    fn lone_pixels_are_absorbed() {
        let mut img = RgbaImage::from_pixel(8, 8, Rgba([50, 50, 50, 255]));
        img.put_pixel(4, 4, Rgba([250, 250, 250, 255]));
        let on = run(&img, &Options { size: 8, colors: 2, clean: true, ..Options::default() }).unwrap();
        assert_eq!(on.get_pixel(4, 4), on.get_pixel(3, 3));
        let off = run(&img, &Options { size: 8, colors: 2, clean: false, ..Options::default() }).unwrap();
        assert_ne!(off.get_pixel(4, 4), off.get_pixel(3, 3));
    }

    #[test]
    fn transparency_survives_and_outline_hugs_the_sprite() {
        let mut img = RgbaImage::new(8, 8);
        for y in 2..6 {
            for x in 2..6 {
                img.put_pixel(x, y, Rgba([200, 100, 50, 255]));
            }
        }
        let plain = run(&img, &Options { size: 8, colors: 2, ..Options::default() }).unwrap();
        assert_eq!(plain.get_pixel(0, 0)[3], 0);
        assert_eq!(plain.get_pixel(1, 3)[3], 0);
        let ringed = run(&img, &Options { size: 8, colors: 2, outline: true, ..Options::default() }).unwrap();
        assert_eq!(ringed.get_pixel(1, 3)[3], 255);
        assert_eq!(ringed.get_pixel(0, 0)[3], 0, "far corners stay clear");
        assert_eq!(ringed.get_pixel(1, 1)[3], 0, "outline is four-connected, no diagonals");
    }

    #[test]
    fn written_png_reads_back_with_its_marks() {
        let img = run(&noisy(), &Options { size: 8, ..Options::default() }).unwrap();
        let path = std::env::temp_dir().join(format!("latent-pixel-test-{}.png", std::process::id()));
        write_png(&path, &img, &serde_json::json!({ "prompt": "kœnig" }), true).unwrap();
        let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap()));
        let reader = dec.read_info().unwrap();
        let info = reader.info();
        assert!(info.uncompressed_latin1_text.iter().any(|t| t.keyword == "ai_generated" && t.text == "true"));
        assert!(info.utf8_text.iter().any(|t| t.keyword == "parameters" && t.get_text().unwrap().contains("kœnig")));
        assert_eq!(image::open(&path).unwrap().to_rgba8(), img);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn erase_fills_from_the_surroundings_with_existing_colors_only() {
        // A left-to-right ramp of 4 tones with a dark blob in the middle.
        let tones: [Rgb; 4] = [[200, 160, 110], [210, 170, 120], [220, 180, 130], [230, 190, 140]];
        let mut img = RgbaImage::new(40, 20);
        for y in 0..20u32 {
            for x in 0..40u32 {
                let c = tones[(x / 10) as usize];
                img.put_pixel(x, y, Rgba([c[0], c[1], c[2], 255]));
            }
        }
        let before = img.clone();
        for y in 8..12 {
            for x in 14..26 {
                img.put_pixel(x, y, Rgba([30, 20, 10, 255]));
            }
        }
        let out = erase(&img, &[[13, 7, 14, 6]]).unwrap();
        // Nothing outside the rectangle changed.
        for y in 0..20u32 {
            for x in 0..40u32 {
                if !((13..27).contains(&x) && (7..13).contains(&y)) {
                    assert_eq!(out.get_pixel(x, y), before.get_pixel(x, y));
                }
            }
        }
        // The blob is gone and only the image's own colors are used.
        for y in 7..13u32 {
            for x in 13..27u32 {
                let c = out.get_pixel(x, y).0;
                assert!(tones.contains(&[c[0], c[1], c[2]]), "({x},{y}) = {c:?}");
            }
        }
        assert!(out.get_pixel(20, 10).0[0] >= 210 && out.get_pixel(20, 10).0[0] <= 230);
    }

    #[test]
    fn a_loose_box_still_takes_the_whole_letter_but_nothing_else() {
        // Paper with a dark bar sticking out of the box on both sides, and a dark thing far away that must stay.
        let mut img = RgbaImage::from_pixel(60, 20, Rgba([220, 180, 130, 255]));
        for x in 20..40 {
            for y in 9..11 {
                img.put_pixel(x, y, Rgba([40, 25, 15, 255]));
            }
        }
        img.put_pixel(55, 3, Rgba([40, 25, 15, 255]));
        let out = erase(&img, &[[23, 7, 14, 6]]).unwrap();
        for x in 20..40u32 {
            for y in 9..11u32 {
                assert_eq!(out.get_pixel(x, y).0, [220, 180, 130, 255], "({x},{y}) should be paper");
            }
        }
        assert_eq!(out.get_pixel(55, 3).0, [40, 25, 15, 255], "far away marks are untouched");
        // Ink 6 pixels past the box is beyond what a box can pull in.
        let mut far = RgbaImage::from_pixel(60, 20, Rgba([220, 180, 130, 255]));
        for x in 20..44 {
            far.put_pixel(x, 10, Rgba([40, 25, 15, 255]));
        }
        let out = erase(&far, &[[24, 7, 8, 6]]).unwrap();
        assert_eq!(out.get_pixel(43, 10).0, [40, 25, 15, 255], "ink far past the box stays");
        assert_eq!(out.get_pixel(30, 10).0, [220, 180, 130, 255]);
    }

    #[test]
    fn erase_clips_ignores_see_through_and_validates() {
        let mut img = RgbaImage::from_pixel(10, 10, Rgba([100, 100, 100, 255]));
        img.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
        assert!(erase(&img, &[[8, 8, 50, 50]]).is_ok(), "an area running off the edge is clipped");
        let out = erase(&img, &[[0, 0, 2, 2]]).unwrap();
        assert_eq!(out.get_pixel(0, 0)[3], 0, "see-through pixels stay see-through");
        assert!(erase(&img, &[]).is_err());
        assert!(erase(&img, &[[50, 50, 5, 5]]).is_err(), "an area off the picture is refused");
        assert!(erase(&RgbaImage::from_pixel(4, 4, Rgba([1, 2, 3, 255])), &[[0, 0, 4, 4]]).is_err(), "nothing left to copy from");
    }

    /// Repeatable pseudo-random value in [0, 1) for building test pictures.
    fn hash(x: i32, y: i32, seed: u32) -> f32 {
        let mut v = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
        v ^= v >> 15;
        v = v.wrapping_mul(0x2C1B_3C6D);
        v ^= v >> 12;
        (v & 0xFFFF) as f32 / 65536.0
    }

    /// Random pixel art of `cols` x `rows` pixels, enlarged by nearest neighbor to `cell` canvas pixels per art pixel.
    fn enlarged(cols: u32, rows: u32, cell: f32) -> RgbaImage {
        let (w, h) = ((cols as f32 * cell).round() as u32, (rows as f32 * cell).round() as u32);
        let mut img = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (cx, cy) = ((x as f32 / cell) as i32, (y as f32 / cell) as i32);
                // Five tones, so about one neighbor in five matches, like real art.
                let v = (hash(cx, cy, 9) * 5.0) as u8 * 50;
                img.put_pixel(x, y, Rgba([v, 255 - v, (v as u32 * 3 % 256) as u8, 255]));
            }
        }
        img
    }

    #[test]
    fn finds_the_pixel_size_of_enlarged_pixel_art() {
        for (cols, rows, cell) in [(40u32, 30u32, 5.0f32), (40, 30, 5.4), (64, 48, 8.0), (30, 30, 12.0), (96, 64, 3.0)] {
            let img = enlarged(cols, rows, cell);
            let found = detect_cell(&img).unwrap_or_else(|| panic!("no grid found for {cols}x{rows} at {cell}"));
            assert!((found - cell).abs() < cell * 0.04, "cell {cell}: found {found}");
            assert_eq!(detect_size(&img), Some(cols.max(rows)), "size for {cols}x{rows} at {cell}");
        }
    }

    #[test]
    fn smooth_pictures_have_no_grid_and_auto_falls_back() {
        let mut img = RgbaImage::new(300, 200);
        for (x, y, p) in img.enumerate_pixels_mut() {
            let v = ((x as f32 / 12.0).sin() * 60.0 + (y as f32 / 17.0).cos() * 50.0 + 128.0) as u8;
            *p = Rgba([v, v / 2 + 40, 255 - v, 255]);
        }
        assert!(detect_cell(&img).is_none());
        let out = run(&img, &Options { size: 0, ..Options::default() }).unwrap();
        assert_eq!(out.width().max(out.height()), 64, "auto without a grid uses 64");
    }

    #[test]
    fn auto_size_rebuilds_the_original_art() {
        let img = enlarged(40, 30, 5.4);
        let out = run(&img, &Options { size: 0, colors: 64, ..Options::default() }).unwrap();
        assert_eq!(out.dimensions(), (40, 30));
    }

    #[test]
    fn small_pixel_art_counts_as_its_own_size() {
        let img = enlarged(40, 30, 1.0);
        assert_eq!(native_size(&img), Some(40));
        let photo = RgbaImage::from_fn(600, 400, |x, y| Rgba([(x % 251) as u8, (y % 241) as u8, ((x * y) % 239) as u8, 255]));
        assert_eq!(native_size(&photo), None);
    }

    #[test]
    fn square_fit_crops_or_pads_for_icons() {
        let img = RgbaImage::from_fn(60, 30, |x, _| if x < 30 { Rgba([200, 0, 0, 255]) } else { Rgba([0, 0, 200, 255]) });
        let crop = run(&img, &Options { size: 16, colors: 4, fit: "crop".into(), ..Options::default() }).unwrap();
        assert_eq!(crop.dimensions(), (16, 16));
        assert!(crop.pixels().all(|p| p[3] == 255), "crop fills the square");
        let pad = run(&img, &Options { size: 16, colors: 4, fit: "pad".into(), ..Options::default() }).unwrap();
        assert_eq!(pad.dimensions(), (16, 16));
        assert_eq!(pad.get_pixel(8, 0)[3], 0, "pad leaves see-through margins");
        assert_eq!(pad.get_pixel(8, 8)[3], 255);
        assert!(run(&img, &Options { fit: "stretch".into(), ..Options::default() }).is_err());
    }

    #[test]
    fn the_keep_pixelated_spec_follows_the_source() {
        // A source that is 32-pixel art in 4 colors, enlarged 8x.
        let mut src = RgbaImage::new(256, 192);
        let tones: [Rgb; 4] = [[10, 10, 10], [200, 50, 50], [50, 200, 50], [240, 240, 100]];
        for (x, y, p) in src.enumerate_pixels_mut() {
            let c = tones[((hash((x / 8) as i32, (y / 8) as i32, 4) * 4.0) as usize).min(3)];
            *p = Rgba([c[0], c[1], c[2], 255]);
        }
        let spec = Spec::default();
        let o = spec.options(Some(&src));
        assert_eq!(o.size, 32, "matches the source's grid");
        assert_eq!(parse_palette(&o.palette).len(), 4, "keeps the source's four colors");
        // An edit rendered big and smooth comes back at that size, in those colors.
        let edit = RgbaImage::from_fn(512, 384, |x, y| Rgba([(x / 3) as u8, (y / 3) as u8, 120, 255]));
        let out = run(&edit, &o).unwrap();
        assert_eq!(out.width().max(out.height()), 32);
        assert!(out.pixels().all(|p| tones.contains(&[p[0], p[1], p[2]])));
        // New images (no source) fall back to 64 and the chosen number of colors.
        assert_eq!(spec.options(None).size, 64);
        assert!(Spec { size: 4, ..Spec::default() }.check().is_err());
        assert!(Spec { size: 16, ..Spec::default() }.check().is_ok());
    }

    #[test]
    fn rejects_out_of_range_options() {
        let img = RgbaImage::new(4, 4);
        assert!(run(&img, &Options { size: 4, ..Options::default() }).is_err());
        assert!(run(&img, &Options { colors: 1, ..Options::default() }).is_err());
    }
}
