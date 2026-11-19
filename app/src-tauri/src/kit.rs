//! The GUI kit: a themed set of Minecraft GUI sprites drawn pixel by pixel, so every edge lands exactly on the grid.
//!
//! Diffusion can't hold a slot grid or a bevel, so the frame of the GUI is drawn here and diffusion (or the pixel tool)
//! is left for ornaments. The look follows the Book of Dragons technique rules:
//!
//! - every material is a **5-tone ramp** built from one base color, shadows leaning cool and highlights warm;
//! - light comes from the **top left**: lit edges above and left, shaded edges below and right;
//! - outlines are a **dark tone of the material**, not black;
//! - corners are **cut**, not square;
//! - grain is **clustered** value noise one tone step apart, never salt and pepper, and there is none unless asked for.

use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};

mod book;

type Rgb = [u8; 3];

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct Options {
    /// `gui` (panel, buttons, slots, bars) or `book` (the open Field Guide: book, torn parchment, rings, swatches, arrows).
    pub kind: String,
    /// `leather`, `crimson`, `parchment`, `iron`, `arcane`, `forest`, or `custom` (then `base` is used). For a book,
    /// the cover.
    pub look: String,
    /// `#rrggbb` for the main material when `look` is `custom`.
    pub base: String,
    /// `#rrggbb` for the glow on hovered buttons and the bar fill. Empty: the look's own.
    pub accent: String,
    /// 0 smooth, 1 light grain, 2 rough grain.
    pub grain: u32,
}

impl Default for Options {
    fn default() -> Self {
        Options { kind: "gui".into(), look: "leather".into(), base: String::new(), accent: String::new(), grain: 1 }
    }
}

#[cfg(test)]
const LOOKS: [&str; 6] = ["leather", "crimson", "parchment", "iron", "arcane", "forest"];

/// (base, accent) of each look.
fn preset(look: &str) -> Option<(Rgb, Rgb)> {
    Some(match look {
        "leather" => ([0x71, 0x3F, 0x28], [0xE0, 0x73, 0x1C]),
        "crimson" => ([0x8E, 0x3A, 0x2A], [0xD8, 0x66, 0x1A]),
        "parchment" => ([0xC9, 0xA5, 0x78], [0xB8, 0x4A, 0x12]),
        "iron" => ([0x6A, 0x72, 0x80], [0x4F, 0xB6, 0xD8]),
        "arcane" => ([0x5A, 0x4A, 0x8C], [0xD0, 0x7A, 0xFF]),
        "forest" => ([0x4E, 0x6B, 0x3A], [0xE0, 0xB8, 0x48]),
        _ => return None,
    })
}

pub fn parse_hex(s: &str) -> Result<Rgb, String> {
    let h = s.trim().trim_start_matches('#');
    if h.len() != 6 || !h.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("\"{s}\" isn't a color. Use #rrggbb."));
    }
    let v = u32::from_str_radix(h, 16).unwrap();
    Ok([(v >> 16) as u8, (v >> 8) as u8, v as u8])
}

impl Options {
    fn colors(&self) -> Result<(Rgb, Rgb), String> {
        if self.grain > 2 {
            return Err("Grain must be 0, 1 or 2.".into());
        }
        let (base, accent) = if self.look == "custom" {
            (parse_hex(&self.base)?, preset("leather").unwrap().1)
        } else {
            preset(&self.look).ok_or_else(|| format!("Unknown look \"{}\".", self.look))?
        };
        let accent = if self.accent.trim().is_empty() { accent } else { parse_hex(&self.accent)? };
        Ok((base, accent))
    }
}

// ---------------------------------------------------------------- color

fn to_hsl(c: Rgb) -> (f32, f32, f32) {
    let (r, g, b) = (c[0] as f32 / 255.0, c[1] as f32 / 255.0, c[2] as f32 / 255.0);
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let l = (mx + mn) / 2.0;
    if mx == mn {
        return (0.0, 0.0, l);
    }
    let d = mx - mn;
    let s = if l > 0.5 { d / (2.0 - mx - mn) } else { d / (mx + mn) };
    let h = if mx == r { (g - b) / d + if g < b { 6.0 } else { 0.0 } } else if mx == g { (b - r) / d + 2.0 } else { (r - g) / d + 4.0 };
    (h * 60.0, s, l)
}

fn from_hsl(h: f32, s: f32, l: f32) -> Rgb {
    let h = h.rem_euclid(360.0) / 360.0;
    let (s, l) = (s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    let f = |mut t: f32| {
        t = t.rem_euclid(1.0);
        let v = if t < 1.0 / 6.0 { p + (q - p) * 6.0 * t } else if t < 0.5 { q } else if t < 2.0 / 3.0 { p + (q - p) * (2.0 / 3.0 - t) * 6.0 } else { p };
        (v * 255.0).round() as u8
    };
    [f(h + 1.0 / 3.0), f(h), f(h - 1.0 / 3.0)]
}

/// Turns `h` toward `target` (degrees) by `amount` (0 to 1), the short way round.
fn toward(h: f32, target: f32, amount: f32) -> f32 {
    let d = (target - h + 540.0).rem_euclid(360.0) - 180.0;
    h + d * amount
}

/// Five tones, dark to light, from `base` as the middle one. Shadows lean toward violet and a touch richer, highlights
/// toward yellow and a touch paler.
#[derive(Clone, Copy)]
pub struct Ramp(pub [Rgb; 5]);

impl Ramp {
    fn new(base: Rgb) -> Ramp {
        let (h, s, l) = to_hsl(base);
        let step = |i: i32| -> Rgb {
            let k = (i - 2) as f32;
            let lightness = (l + [-0.25, -0.12, 0.0, 0.11, 0.2][i as usize]).clamp(0.05, 0.95);
            let hue = if i < 2 { toward(h, 270.0, 0.07 * -k) } else if i > 2 { toward(h, 50.0, 0.08 * k) } else { h };
            let sat = if i < 2 { s * (1.0 + 0.08 * -k) } else { s * (1.0 - 0.1 * k) };
            from_hsl(hue, sat, lightness)
        };
        Ramp([step(0), step(1), step(2), step(3), step(4)])
    }

    /// The same material one notch brighter, for hover.
    fn lifted(self, base: Rgb) -> Ramp {
        let (h, s, l) = to_hsl(base);
        Ramp::new(from_hsl(h, s, l + 0.08))
    }

    /// The same material worn out: grey and dim, for disabled.
    fn faded(base: Rgb) -> Ramp {
        let (h, s, l) = to_hsl(base);
        Ramp::new(from_hsl(h, s * 0.3, l * 0.8))
    }

    /// Darker than the darkest tone: the outline.
    fn outline(&self) -> Rgb {
        let (h, s, l) = to_hsl(self.0[0]);
        from_hsl(toward(h, 270.0, 0.12), s, l * 0.55)
    }
}

// ---------------------------------------------------------------- drawing

/// Cheap deterministic hash to [0, 1).
fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut v = (x as u32).wrapping_mul(0x9E37_79B1) ^ (y as u32).wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
    v ^= v >> 15;
    v = v.wrapping_mul(0x2C1B_3C6D);
    v ^= v >> 12;
    v = v.wrapping_mul(0x297A_2D39);
    v ^= v >> 15;
    (v & 0xFFFF) as f32 / 65536.0
}

/// Clustered value noise in [0, 1): two shifted coarse grids averaged, so patches are blobs a few pixels wide.
fn cluster(x: u32, y: u32, seed: u32) -> f32 {
    let (x, y) = (x as i32, y as i32);
    (hash(x.div_euclid(3), y.div_euclid(3), seed) + hash((x + 1).div_euclid(2), (y + 2).div_euclid(3), seed ^ 0x55)) / 2.0
}

/// Whether (x, y) is inside a `w` x `h` shape whose four corners are cut by `cut` pixels (0 square, 1 notch, 2 stepped).
fn inside(x: u32, y: u32, w: u32, h: u32, cut: u32) -> bool {
    if x >= w || y >= h {
        return false;
    }
    let dx = x.min(w - 1 - x);
    let dy = y.min(h - 1 - y);
    dx + dy >= cut
}

#[derive(Clone, Copy, PartialEq)]
enum Relief {
    Raised,
    Sunken,
}

struct Piece {
    w: u32,
    h: u32,
    cut: u32,
    relief: Relief,
    ramp: Ramp,
    outline: Rgb,
    /// 0 to 2.
    grain: u32,
    seed: u32,
    /// An irregular shape (`w * h` flags, row by row) instead of the rectangle with cut corners.
    mask: Option<Vec<bool>>,
}

impl Piece {
    fn draw(&self) -> RgbaImage {
        let (w, h) = (self.w, self.h);
        let mut img = RgbaImage::new(w, h);
        let t = self.ramp.0;
        let out_dark = self.outline;
        let solid = |x: i64, y: i64| {
            x >= 0
                && y >= 0
                && match &self.mask {
                    Some(m) => (x as u32) < w && (y as u32) < h && m[(y as u32 * w + x as u32) as usize],
                    None => inside(x as u32, y as u32, w, h, self.cut),
                }
        };
        // Outline ring: inside pixels that touch the outside (or the edge) in a cardinal direction.
        let is_outline = |x: u32, y: u32| {
            let (xi, yi) = (x as i64, y as i64);
            solid(xi, yi) && !(solid(xi - 1, yi) && solid(xi + 1, yi) && solid(xi, yi - 1) && solid(xi, yi + 1))
        };
        let (lit, shade) = match self.relief {
            Relief::Raised => (t[3], t[1]),
            Relief::Sunken => (t[1], t[3]),
        };
        for y in 0..h {
            for x in 0..w {
                if !solid(x as i64, y as i64) {
                    continue;
                }
                let (xi, yi) = (x as i64, y as i64);
                let c = if is_outline(x, y) {
                    // The lit side of the outline is a touch lighter than the shaded side.
                    if (!solid(xi, yi - 1) || !solid(xi - 1, yi)) && solid(xi + 1, yi) && solid(xi, yi + 1) { mix(out_dark, t[0], 0.5) } else { out_dark }
                } else if is_outline(x, y.wrapping_sub(1)) && y > 0 || is_outline(x.wrapping_sub(1), y) && x > 0 {
                    lit
                } else if is_outline(x, y + 1) || is_outline(x + 1, y) {
                    shade
                } else {
                    self.fill(x, y)
                };
                img.put_pixel(x, y, Rgba([c[0], c[1], c[2], 255]));
            }
        }
        img
    }

    /// The face: the middle tone, with sparse clusters one step lighter or darker when there's grain.
    fn fill(&self, x: u32, y: u32) -> Rgb {
        let t = self.ramp.0;
        let base = if self.relief == Relief::Sunken { t[1] } else { t[2] };
        if self.grain == 0 {
            return base;
        }
        let n = cluster(x, y, self.seed);
        let (lo, hi) = if self.grain == 1 { (0.22, 0.78) } else { (0.32, 0.68) };
        // The middle of a nine-slice is stretched by the game, so patches are kept soft: a fraction of a step, not a whole one.
        let k = if self.grain == 1 { 0.4 } else { 0.7 };
        let up = mix(base, if self.relief == Relief::Sunken { t[2] } else { t[3] }, k);
        let down = mix(base, if self.relief == Relief::Sunken { t[0] } else { t[1] }, k);
        if n < lo { down } else if n > hi { up } else { base }
    }
}

fn mix(a: Rgb, b: Rgb, t: f32) -> Rgb {
    [0, 1, 2].map(|i| (a[i] as f32 + (b[i] as f32 - a[i] as f32) * t).round() as u8)
}

fn put(img: &mut RgbaImage, x: u32, y: u32, c: Rgb) {
    img.put_pixel(x, y, Rgba([c[0], c[1], c[2], 255]));
}

/// A recessed groove with a small bevel (the bar's empty track).
fn groove(w: u32, h: u32, ramp: Ramp) -> RgbaImage {
    Piece { w, h, cut: 1, relief: Relief::Sunken, ramp, outline: ramp.outline(), grain: 0, seed: 0, mask: None }.draw()
}

/// The bar's fill: an accent ramp inside a dark rim, bright along the top, with a hot spot at the lit end.
fn fill_bar(w: u32, h: u32, accent: Ramp) -> RgbaImage {
    let mut img = Piece { w, h, cut: 1, relief: Relief::Raised, ramp: accent, outline: accent.outline(), grain: 0, seed: 0, mask: None }.draw();
    let t = accent.0;
    let rows = h.saturating_sub(2);
    for i in 0..rows {
        // Bright at the top, darker toward the bottom.
        let tone = match (i * 5 / rows.max(1)).min(4) { 0 => t[4], 1 => t[3], 2 | 3 => t[2], _ => t[1] };
        for x in 1..w - 1 {
            if img.get_pixel(x, i + 1)[3] != 0 {
                put(&mut img, x, i + 1, tone);
            }
        }
    }
    if w > 4 && h > 3 {
        put(&mut img, 2, 1, mix(t[4], [255, 255, 255], 0.5));
    }
    img
}

// ---------------------------------------------------------------- the kit

pub struct Sprite {
    /// File name under the folder: `panel`, `button_hover`...
    pub id: &'static str,
    pub label: &'static str,
    pub image: RgbaImage,
    /// `nine_slice` or `stretch`.
    pub scaling: &'static str,
    /// Left, top, right, bottom.
    pub border: [u32; 4],
}

/// Every sprite of the kit, plus a preview that assembles them the way the game would.
pub fn build(o: &Options) -> Result<(Vec<Sprite>, RgbaImage), String> {
    let (base, accent) = o.colors()?;
    match o.kind.as_str() {
        "gui" => {}
        "book" => return book::build(o, base, accent),
        other => return Err(format!("Unknown kit \"{other}\".")),
    }
    let ramp = Ramp::new(base);
    let hover = ramp.lifted(base);
    let faded = Ramp::faded(base);
    let acc = Ramp::new(accent);
    let piece = |w, h, cut, relief, ramp: Ramp, grain, seed| Piece { w, h, cut, relief, ramp, outline: ramp.outline(), grain, seed, mask: None }.draw();

    let panel = piece(32, 32, 2, Relief::Raised, ramp, o.grain, 11);
    let inset = piece(32, 32, 1, Relief::Sunken, ramp, o.grain.min(1), 23);
    // Slots sit deeper than the inset around them, so they read as holes.
    let (bh, bs, bl) = to_hsl(base);
    let slot = piece(18, 18, 0, Relief::Sunken, Ramp::new(from_hsl(bh, bs, bl - 0.07)), 0, 37);
    let button = piece(24, 20, 1, Relief::Raised, ramp, 0, 41);
    let mut button_hover = piece(24, 20, 1, Relief::Raised, hover, 0, 43);
    // Hover glows: the rim takes the accent.
    let glow = acc.0[2];
    for (x, y, p) in button_hover.enumerate_pixels_mut() {
        if p[3] != 0 && (x == 0 || y == 0 || x == 23 || y == 19) {
            *p = Rgba([glow[0], glow[1], glow[2], 255]);
        }
    }
    let button_disabled = piece(24, 20, 1, Relief::Raised, faded, 0, 47);
    let bar_bg = groove(32, 8, ramp);
    let bar_fill = fill_bar(32, 8, acc);

    let sprites = vec![
        Sprite { id: "panel", label: "Panel", image: panel, scaling: "nine_slice", border: [5, 5, 5, 5] },
        Sprite { id: "inset", label: "Inset", image: inset, scaling: "nine_slice", border: [4, 4, 4, 4] },
        Sprite { id: "slot", label: "Slot", image: slot, scaling: "stretch", border: [0; 4] },
        Sprite { id: "button", label: "Button", image: button, scaling: "nine_slice", border: [4, 4, 4, 4] },
        Sprite { id: "button_hover", label: "Button hover", image: button_hover, scaling: "nine_slice", border: [4, 4, 4, 4] },
        Sprite { id: "button_disabled", label: "Button disabled", image: button_disabled, scaling: "nine_slice", border: [4, 4, 4, 4] },
        Sprite { id: "bar_bg", label: "Bar track", image: bar_bg, scaling: "nine_slice", border: [3, 3, 3, 3] },
        Sprite { id: "bar_fill", label: "Bar fill", image: bar_fill, scaling: "nine_slice", border: [3, 3, 3, 3] },
    ];
    let preview = compose(&sprites);
    Ok((sprites, preview))
}

/// Draws `src` into `dst` at (x, y), `w` x `h`, the way the game's nine-slice does: corners as they are, edges and
/// middle stretched.
pub fn nine_slice(dst: &mut RgbaImage, src: &RgbaImage, border: [u32; 4], x: u32, y: u32, w: u32, h: u32) {
    let [l, t, r, b] = border;
    let (sw, sh) = src.dimensions();
    // Source and destination column / row ranges: (source start, source length, dest start, dest length).
    let cols = [(0, l, 0, l), (l, sw - l - r, l, w - l - r), (sw - r, r, w - r, r)];
    let rows = [(0, t, 0, t), (t, sh - t - b, t, h - t - b), (sh - b, b, h - b, b)];
    for &(sy0, sl, dy0, dl) in &rows {
        for &(sx0, sw_, dx0, dw) in &cols {
            for dy in 0..dl {
                for dx in 0..dw {
                    let sx = sx0 + if dw == 0 || sw_ == 0 { 0 } else { dx * sw_ / dw };
                    let sy = sy0 + if dl == 0 || sl == 0 { 0 } else { dy * sl / dl };
                    let p = *src.get_pixel(sx.min(sw - 1), sy.min(sh - 1));
                    if p[3] != 0 && x + dx0 + dx < dst.width() && y + dy0 + dy < dst.height() {
                        dst.put_pixel(x + dx0 + dx, y + dy0 + dy, p);
                    }
                }
            }
        }
    }
}

/// A mock-up screen at its real pixel size: a panel with an inset, a slot grid, three buttons and a half-full bar.
fn compose(s: &[Sprite]) -> RgbaImage {
    let get = |id: &str| s.iter().find(|p| p.id == id).unwrap();
    let draw = |img: &mut RgbaImage, id: &str, x, y, w, h| {
        let p = get(id);
        if p.scaling == "nine_slice" { nine_slice(img, &p.image, p.border, x, y, w, h) } else { nine_slice(img, &p.image, [0; 4], x, y, w, h) }
    };
    let (w, h) = (136, 92);
    let mut img = RgbaImage::new(w, h);
    // The backdrop is the panel's darkest tone, so the cut corners read.
    let back = get("panel").image.get_pixel(1, 1).0;
    for p in img.pixels_mut() {
        *p = Rgba([back[0] / 4, back[1] / 4, back[2] / 4, 255]);
    }
    draw(&mut img, "panel", 4, 4, 128, 84);
    draw(&mut img, "inset", 10, 10, 66, 40);
    for row in 0..2 {
        for col in 0..3 {
            draw(&mut img, "slot", 14 + col * 20, 14 + row * 20, 18, 18);
        }
    }
    draw(&mut img, "button", 82, 12, 44, 20);
    draw(&mut img, "button_hover", 82, 36, 44, 20);
    draw(&mut img, "button_disabled", 82, 60, 44, 20);
    draw(&mut img, "bar_bg", 10, 58, 66, 8);
    draw(&mut img, "bar_fill", 10, 58, 36, 8);
    draw(&mut img, "bar_bg", 10, 70, 66, 8);
    draw(&mut img, "bar_fill", 10, 70, 56, 8);
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_get_lighter_and_shift_hue() {
        for look in LOOKS {
            let (base, _) = preset(look).unwrap();
            let r = Ramp::new(base).0;
            for i in 0..4 {
                assert!(luma(r[i]) < luma(r[i + 1]), "{look}: tones must rise in brightness");
            }
            assert!(luma(Ramp::new(base).outline()) < luma(r[0]), "{look}: outline darker than the ramp");
        }
        // Warm highlight, cool shadow: a brown's lightest tone has more red-over-blue than its darkest.
        let r = Ramp::new([0x71, 0x3F, 0x28]).0;
        let warmth = |c: Rgb| c[0] as i32 - c[2] as i32;
        assert!(warmth(r[4]) > warmth(r[0]));
    }

    fn luma(c: Rgb) -> i32 {
        299 * c[0] as i32 + 587 * c[1] as i32 + 114 * c[2] as i32
    }

    #[test]
    fn every_sprite_has_its_size_a_cut_corner_and_a_dark_outline() {
        let (sprites, _) = build(&Options::default()).unwrap();
        assert_eq!(sprites.len(), 8);
        for s in &sprites {
            let [l, t, r, b] = s.border;
            let (w, h) = s.image.dimensions();
            assert!(l + r < w && t + b < h || s.scaling == "stretch", "{}: border must leave a middle", s.id);
            // The outline ring is fully opaque.
            for x in 2..w - 2 {
                assert_eq!(s.image.get_pixel(x, 0)[3], 255, "{} top edge", s.id);
            }
        }
        let panel = &sprites[0].image;
        assert_eq!(panel.get_pixel(0, 0)[3], 0, "a cut corner leaves the corner pixel clear");
        assert_ne!(panel.get_pixel(16, 16), panel.get_pixel(16, 0), "face and outline differ");
        let slot = &sprites[2].image;
        assert_eq!(slot.get_pixel(0, 0)[3], 255, "slots are square");
    }

    #[test]
    fn lit_from_the_top_left() {
        let (sprites, _) = build(&Options { grain: 0, ..Options::default() }).unwrap();
        let p = &sprites[0].image;
        assert!(luma(p.get_pixel(16, 1).0[..3].try_into().unwrap()) > luma(p.get_pixel(16, 30).0[..3].try_into().unwrap()));
        assert!(luma(p.get_pixel(1, 16).0[..3].try_into().unwrap()) > luma(p.get_pixel(30, 16).0[..3].try_into().unwrap()));
        // Sunken is the other way round.
        let i = &sprites[1].image;
        assert!(luma(i.get_pixel(16, 1).0[..3].try_into().unwrap()) < luma(i.get_pixel(16, 30).0[..3].try_into().unwrap()));
    }

    #[test]
    fn grain_is_clustered_and_only_when_asked() {
        let colors = |grain| {
            let (s, _) = build(&Options { grain, ..Options::default() }).unwrap();
            let img = &s[0].image;
            let mut set = std::collections::HashSet::new();
            for y in 3..29 {
                for x in 3..29 {
                    set.insert(img.get_pixel(x, y).0);
                }
            }
            (set.len(), img.clone())
        };
        assert_eq!(colors(0).0, 1, "smooth: one flat tone in the face");
        let (n, img) = colors(2);
        assert_eq!(n, 3, "grain stays within one step either side");
        // Clustered: most pixels match the one to their right (salt and pepper would not).
        let mut same = 0;
        for y in 3..29 {
            for x in 3..28 {
                same += (img.get_pixel(x, y) == img.get_pixel(x + 1, y)) as u32;
            }
        }
        assert!(same as f32 / (26.0 * 25.0) > 0.6, "grain should form patches, got {same}");
    }

    #[test]
    fn nine_slice_keeps_corners_and_grows() {
        let (sprites, _) = build(&Options::default()).unwrap();
        let p = &sprites[0];
        let mut big = RgbaImage::new(90, 60);
        nine_slice(&mut big, &p.image, p.border, 0, 0, 90, 60);
        assert_eq!(big.get_pixel(0, 0)[3], 0);
        assert_eq!(big.get_pixel(1, 2), p.image.get_pixel(1, 2));
        assert_eq!(big.get_pixel(88, 57), p.image.get_pixel(30, 29));
        assert!(big.pixels().all(|q| q[3] == 255 || q[3] == 0));
        assert_eq!(big.get_pixel(45, 0), p.image.get_pixel(16, 0), "top edge stretches along the outline");
    }

    #[test]
    fn custom_look_and_accent_parse() {
        let o = Options { look: "custom".into(), base: "#336699".into(), accent: "ff0000".into(), grain: 0, ..Options::default() };
        assert!(build(&o).is_ok());
        assert!(build(&Options { look: "custom".into(), base: "blue".into(), ..Options::default() }).is_err());
        assert!(build(&Options { look: "nope".into(), ..Options::default() }).is_err());
        assert!(build(&Options { grain: 3, ..Options::default() }).is_err());
        assert!(build(&Options { accent: "#12".into(), ..Options::default() }).is_err());
    }

    #[test]
    fn hover_rim_takes_the_accent_and_disabled_is_duller() {
        let (s, _) = build(&Options::default()).unwrap();
        let rim = s[4].image.get_pixel(12, 0).0;
        let plain = s[3].image.get_pixel(12, 0).0;
        assert_ne!(rim, plain);
        let sat = |c: [u8; 4]| to_hsl([c[0], c[1], c[2]]).1;
        assert!(sat(s[5].image.get_pixel(12, 10).0) < sat(s[3].image.get_pixel(12, 10).0));
    }
}
