//! The Field Guide kit: an open book you can put controls on, drawn without any text (Minecraft draws its own).
//!
//! Sprites: the book itself (cover, page stack, torn page tops, gutter shading, ring binding, ribbon), a torn parchment
//! inset that stretches, a brush-stroke label, a swatch frame, arrows, a row divider and an ink rule.

use super::*;

const INK: Rgb = [0x3B, 0x29, 0x1E];
const PAGE: Rgb = [0xE6, 0xC4, 0x92];
const BRASS: Rgb = [0xC9, 0x8F, 0x34];
const W: u32 = 256;
const H: u32 = 168;
/// Page area inside the cover.
const PX0: u32 = 7;
const PX1: u32 = W - 8;
const PY0: u32 = 7;
const PYB: u32 = H - 10;
const MID: u32 = W / 2;

// ---------------------------------------------------------------- shapes

fn noise1(x: i32, period: i32, seed: u32) -> f32 {
    let (i, f) = (x.div_euclid(period), x.rem_euclid(period) as f32 / period as f32);
    let (a, b) = (hash(i, 0, seed), hash(i + 1, 0, seed));
    a + (b - a) * f * f * (3.0 - 2.0 * f)
}

/// How far a torn edge bites in at position `x` along it: 0 to `amp` pixels, in short irregular runs.
fn tear(x: i32, seed: u32, amp: i32) -> i32 {
    let n = noise1(x, 7, seed) * 0.65 + noise1(x, 3, seed ^ 0x9) * 0.35;
    ((n * (amp as f32 + 0.999)) as i32).clamp(0, amp)
}

/// A `w` x `h` rectangle whose four edges are torn.
fn torn_mask(w: u32, h: u32, amp: i32, seed: u32) -> Vec<bool> {
    let mut m = vec![false; (w * h) as usize];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            m[(y as u32 * w + x as u32) as usize] = x >= tear(y, seed + 1, amp)
                && x < w as i32 - tear(y, seed + 2, amp)
                && y >= tear(x, seed + 3, amp)
                && y < h as i32 - tear(x, seed + 4, amp);
        }
    }
    m
}

/// 4 x 4 ordered-dither threshold in (0, 1).
fn bayer(x: u32, y: u32) -> f32 {
    const M: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];
    (M[(y % 4) as usize][(x % 4) as usize] as f32 + 0.5) / 16.0
}

/// A ramp index from a continuous position on the ramp (0 darkest, 4 lightest), dithered between neighbors.
fn tone_index(v: f32, x: u32, y: u32) -> usize {
    let v = v.clamp(0.0, 4.0);
    let lo = v.floor();
    (lo as usize + ((v - lo) > bayer(x, y)) as usize).min(4)
}

// ---------------------------------------------------------------- the book

/// Parchment: warm tan edges to cream, kept low in saturation. (The generic ramp turns a pale base into orange.)
fn page_ramp() -> Ramp {
    Ramp([[0x8B, 0x5A, 0x36], [0xC4, 0x92, 0x5E], [0xE6, 0xC4, 0x92], [0xF2, 0xD8, 0xAA], [0xFB, 0xEA, 0xC8]])
}

fn ring(brass: &Ramp, hole: Rgb) -> RgbaImage {
    let mut img = RgbaImage::new(10, 7);
    for y in 0..7u32 {
        for x in 0..10u32 {
            let (nx, ny) = ((x as f32 - 4.5) / 4.8, (y as f32 - 3.0) / 3.3);
            let r2 = nx * nx + ny * ny;
            if r2 > 1.0 {
                continue;
            }
            let c = if r2 > 0.8 {
                brass.0[0]
            } else if r2 > 0.36 {
                if ny < -0.1 { brass.0[4] } else if ny > 0.35 { brass.0[1] } else { brass.0[2] }
            } else {
                hole
            };
            put(&mut img, x, y, c);
        }
    }
    img
}

fn ribbon_pixel(x: u32, y: u32, r: &Ramp) -> Option<Rgb> {
    let (x0, top, bottom) = (MID - 3, H - 28, H - 3);
    if x < x0 || x >= x0 + 7 || y < top || y > bottom {
        return None;
    }
    // Swallowtail: the bottom edge bites in.
    let dx = (x as i32 - MID as i32).abs();
    let up = (bottom - y) as i32;
    if up < 3 && dx <= 2 - up {
        return None;
    }
    let edge = x == x0 || x == x0 + 6 || y == top;
    Some(if edge {
        r.outline()
    } else if x == x0 + 1 {
        r.0[3]
    } else if x >= x0 + 5 {
        r.0[1]
    } else {
        r.0[2]
    })
}

fn book_bg(cover: &Ramp, page: &Ramp, brass: &Ramp, ribbon: &Ramp, grain: u32) -> RgbaImage {
    let mut img = Piece { w: W, h: H, cut: 4, relief: Relief::Raised, ramp: *cover, outline: cover.outline(), grain: grain.max(1), seed: 5, mask: None }.draw();

    // Stitching along the cover's edge.
    let stitch = mix(cover.0[3], page.0[3], 0.45);
    for x in 5..W - 5 {
        if x % 4 < 2 {
            put(&mut img, x, 4, stitch);
            put(&mut img, x, H - 5, stitch);
        }
    }
    for y in 5..H - 5 {
        if y % 4 < 2 {
            put(&mut img, 4, y, stitch);
            put(&mut img, W - 5, y, stitch);
        }
    }

    // The page block's edges, stacked under the pages.
    for r in 0..7u32 {
        let y = PYB + 1 + r;
        let inset = if r % 3 == 2 { 1 } else { 0 };
        for x in PX0 + inset..=PX1 - inset {
            let gap = hash(x as i32, r as i32, 61) > 0.93;
            let c = if r % 2 == 1 { page.0[0] } else if gap { page.0[1] } else { page.0[2] };
            put(&mut img, x, y, if r == 6 { page.outline() } else { c });
        }
    }

    // The two pages: torn at the top and bottom, shaded toward their outer edges and the gutter.
    for (xa, xb) in [(PX0, MID - 4), (MID + 3, PX1)] {
        for x in xa..=xb {
            let top = PY0 as i32 + tear(x as i32, 91, 2);
            let bot = PYB as i32 - tear(x as i32, 77, 1);
            for y in top..=bot {
                let (xi, yi) = (x as i32, y);
                let d_out = (xi - xa as i32).min(xb as i32 - xi).min(yi - top).min(bot - yi);
                let d_gut = (xi - MID as i32).abs();
                let edge_shade = (1.0 - d_out as f32 / 4.0).clamp(0.0, 1.0) * 0.7;
                let gut_shade = (1.0 - (d_gut - 3) as f32 / 14.0).clamp(0.0, 1.0);
                let s = edge_shade.max(gut_shade);
                let v = 3.0 - s * 2.4;
                let i = tone_index(v, x, y as u32);
                let c = if d_out == 0 { page.0[0] } else { page.0[i] };
                put(&mut img, x, y as u32, c);
            }
        }
    }
    // Gutter: the crease between the pages.
    for y in PY0 + 1..=PYB {
        for x in MID - 3..MID + 3 {
            let c = if x == MID - 1 || x == MID { page.0[0] } else { page.0[1] };
            if img.get_pixel(x, y)[3] != 0 {
                put(&mut img, x, y, c);
            }
        }
    }

    // Ring binding across the gutter.
    let r = ring(brass, page.0[0]);
    for cy in [20u32, 47, 74, 101, 128] {
        for (x, y, p) in r.enumerate_pixels() {
            if p[3] != 0 {
                img.put_pixel(MID - 5 + x, cy - 3 + y, *p);
            }
        }
    }

    // Ribbon bookmark.
    for y in 0..H {
        for x in 0..W {
            if let Some(c) = ribbon_pixel(x, y, ribbon) {
                put(&mut img, x, y, c);
            }
        }
    }
    img
}

// ---------------------------------------------------------------- the small parts

fn torn_inset(page: &Ramp) -> RgbaImage {
    // One notch lighter than the page, and smooth: the game stretches the middle, so any grain would smear into blocks.
    let lifted = Ramp([page.0[1], page.0[2], page.0[3], page.0[4], page.0[4]]);
    let outline = mix(page.0[0], page.0[1], 0.45);
    Piece { w: 32, h: 32, cut: 0, relief: Relief::Raised, ramp: lifted, outline, grain: 0, seed: 29, mask: Some(torn_mask(32, 32, 3, 100)) }.draw()
}

/// The torn parchment as a stretched middle with soft grain; used for the big page window.
fn brush_label(page: &Ramp) -> RgbaImage {
    let (w, h) = (40u32, 14u32);
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        let (l, r) = (tear(y as i32, 120, 4), tear(y as i32, 121, 4));
        for x in 0..w {
            let ragged = y == 0 && hash(x as i32, 0, 5) > 0.8 || y == h - 1 && hash(x as i32, 1, 5) > 0.75;
            if x as i32 >= l && (x as i32) < w as i32 - r && !ragged {
                // A brush stroke: darker tan with a few lighter streaks along it.
                let streak = hash(x as i32 / 6, y as i32 / 3, 8) > 0.78;
                let c = if y == 0 || y == h - 1 { mix(page.0[0], page.0[1], 0.4) } else if streak { page.0[2] } else if y < 4 { page.0[1] } else { mix(page.0[1], page.0[2], 0.3) };
                put(&mut img, x, y, c);
            }
        }
    }
    img
}

fn swatch() -> RgbaImage {
    let (w, h) = (14u32, 11u32);
    let mut img = RgbaImage::new(w, h);
    // Pale so the game can tint it with a color. The shape leans right by one pixel and has a torn corner or two.
    let shape = |x: i32, y: i32| {
        let lean = if y < 5 { 1 } else { 0 };
        let (l, r) = (lean, w as i32 - 2 + lean);
        x >= l && x <= r && y >= 0 && y < h as i32 && !(x == r && y == 0) && !(x == l && y == h as i32 - 1)
    };
    let (edge, lit, face, shade) = (mix(INK, [0xD6, 0xD2, 0xCA], 0.45), [0xF1, 0xEE, 0xE6], [0xD9, 0xD5, 0xCD], [0xB4, 0xAF, 0xA6]);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            if !shape(x, y) {
                continue;
            }
            let rim = !shape(x - 1, y) || !shape(x + 1, y) || !shape(x, y - 1) || !shape(x, y + 1);
            let c = if rim { edge } else if !shape(x - 1, y - 1) || !shape(x, y - 1) || !shape(x - 1, y) { lit } else if !shape(x + 1, y + 1) || !shape(x, y + 1) || !shape(x + 1, y) { shade } else { face };
            put(&mut img, x as u32, y as u32, c);
        }
    }
    img
}

fn arrow(left: bool, color: Rgb) -> RgbaImage {
    let mut img = RgbaImage::new(7, 11);
    for y in 0..11u32 {
        let d = (y as i32 - 5).unsigned_abs();
        let x = if left { d } else { 6 - d - 1 };
        put(&mut img, x, y, color);
        put(&mut img, x + 1, y, color);
    }
    img
}

fn divider(page: &Ramp) -> RgbaImage {
    let mut img = RgbaImage::new(32, 2);
    for x in 2..30 {
        put(&mut img, x, 0, page.0[1]);
        put(&mut img, x, 1, page.0[3]);
    }
    img
}

/// A broken double rule in ink, like the headings in the reference book: a steady line and a shorter, lighter dash
/// above it.
fn rule() -> RgbaImage {
    let mut img = RgbaImage::new(96, 5);
    for x in 2..94u32 {
        put(&mut img, x, 3, INK);
    }
    for x in 22..60u32 {
        put(&mut img, x, 0, mix(INK, PAGE, 0.4));
    }
    img
}

// ---------------------------------------------------------------- the kit

pub fn build(o: &Options, cover_base: Rgb, accent: Rgb) -> Result<(Vec<Sprite>, RgbaImage), String> {
    let cover = Ramp::new(cover_base);
    let page = page_ramp();
    let brass = Ramp::new(BRASS);
    let ribbon = Ramp::new(accent);

    let sprites = vec![
        Sprite { id: "book", label: "Book", image: book_bg(&cover, &page, &brass, &ribbon, o.grain), scaling: "stretch", border: [0; 4] },
        Sprite { id: "inset_torn", label: "Torn page", image: torn_inset(&page), scaling: "nine_slice", border: [8, 8, 8, 8] },
        Sprite { id: "label_brush", label: "Brush label", image: brush_label(&page), scaling: "nine_slice", border: [6, 4, 6, 4] },
        Sprite { id: "swatch", label: "Swatch", image: swatch(), scaling: "stretch", border: [0; 4] },
        Sprite { id: "arrow_left", label: "Arrow left", image: arrow(true, INK), scaling: "stretch", border: [0; 4] },
        Sprite { id: "arrow_right", label: "Arrow right", image: arrow(false, INK), scaling: "stretch", border: [0; 4] },
        Sprite { id: "arrow_left_hover", label: "Arrow left hover", image: arrow(true, ribbon.0[2]), scaling: "stretch", border: [0; 4] },
        Sprite { id: "arrow_right_hover", label: "Arrow right hover", image: arrow(false, ribbon.0[2]), scaling: "stretch", border: [0; 4] },
        Sprite { id: "divider", label: "Divider", image: divider(&page), scaling: "stretch", border: [0; 4] },
        Sprite { id: "rule", label: "Ink rule", image: rule(), scaling: "stretch", border: [0; 4] },
    ];
    let preview = compose(&sprites);
    Ok((sprites, preview))
}

/// The Field Guide screen as laid out in the reference: a window on the left page, label rows with swatches and arrows
/// on the right, rules under both titles.
fn compose(s: &[Sprite]) -> RgbaImage {
    let get = |id: &str| s.iter().find(|p| p.id == id).unwrap();
    let draw = |img: &mut RgbaImage, id: &str, x: u32, y: u32, w: u32, h: u32| {
        let p = get(id);
        nine_slice(img, &p.image, if p.scaling == "nine_slice" { p.border } else { [0; 4] }, x, y, w, h);
    };
    let mut img = RgbaImage::new(W, H);
    for p in img.pixels_mut() {
        *p = Rgba([16, 12, 12, 255]);
    }
    let book = &get("book").image;
    for (x, y, p) in book.enumerate_pixels() {
        if p[3] != 0 {
            img.put_pixel(x, y, *p);
        }
    }
    draw(&mut img, "rule", 14, 20, 104, 5);
    draw(&mut img, "rule", 138, 20, 104, 5);
    draw(&mut img, "label_brush", 14, 28, 52, 14);
    draw(&mut img, "inset_torn", 12, 46, 110, 96);
    draw(&mut img, "inset_torn", 134, 30, 110, 112);
    for i in 0..5u32 {
        let y = 36 + i * 21;
        draw(&mut img, "swatch", 142, y + 4, 14, 11);
        draw(&mut img, "arrow_left", 188, y + 4, 7, 11);
        draw(&mut img, "arrow_right", 228, y + 4, 7, 11);
        if i < 4 {
            draw(&mut img, "divider", 140, y + 19, 98, 2);
        }
    }
    for i in 0..5u32 {
        draw(&mut img, "swatch", 32 + i * 18, 148, 14, 11);
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book() -> (Vec<Sprite>, RgbaImage) {
        let o = Options { kind: "book".into(), look: "crimson".into(), ..Options::default() };
        build(&o, preset("crimson").unwrap().0, preset("crimson").unwrap().1).unwrap()
    }

    #[test]
    fn tear_stays_within_its_amplitude_and_varies() {
        let v: Vec<i32> = (0..200).map(|x| tear(x, 3, 2)).collect();
        assert!(v.iter().all(|t| (0..=2).contains(t)));
        assert!(v.iter().any(|t| *t == 0) && v.iter().any(|t| *t == 2));
    }

    #[test]
    fn the_book_has_its_cover_ring_binding_and_ribbon() {
        let (sprites, preview) = book();
        let b = &sprites[0].image;
        assert_eq!(b.dimensions(), (W, H));
        assert_eq!(b.get_pixel(0, 0)[3], 0, "the cover's corner is cut");
        assert_eq!(b.get_pixel(W / 2, 20)[3], 255, "a ring sits in the gutter");
        assert_eq!(b.get_pixel(MID, H - 10)[3], 255);
        assert!(ribbon_pixel(MID, H - 10, &Ramp::new([200, 100, 30])).is_some());
        assert!(ribbon_pixel(MID, H - 3, &Ramp::new([200, 100, 30])).is_none(), "the ribbon's tail is notched");
        assert_eq!(preview.dimensions(), (W, H));
    }

    #[test]
    fn every_book_sprite_is_usable() {
        let (sprites, _) = book();
        assert_eq!(sprites.len(), 10);
        for s in &sprites {
            let [l, t, r, b] = s.border;
            let (w, h) = s.image.dimensions();
            assert!(s.scaling == "stretch" || (l + r < w && t + b < h && l > 0 && t > 0), "{}", s.id);
            assert!(s.image.pixels().any(|p| p[3] == 255), "{} is empty", s.id);
        }
        let inset = &sprites[1].image;
        // The torn edge leaves the outer ring ragged, so it isn't a clean rectangle.
        let opaque_top: Vec<_> = (0..32).map(|x| (0..4).find(|y| inset.get_pixel(x, *y)[3] != 0).unwrap_or(9)).collect();
        assert!(opaque_top.iter().min() != opaque_top.iter().max(), "the top edge is torn");
    }

    #[test]
    fn pages_are_lighter_in_the_middle_than_by_the_gutter() {
        let (sprites, _) = book();
        let b = &sprites[0].image;
        let l = |x: u32, y: u32| {
            let p = b.get_pixel(x, y).0;
            299 * p[0] as i32 + 587 * p[1] as i32 + 114 * p[2] as i32
        };
        let mut mid = 0;
        let mut gutter = 0;
        for y in 60..100 {
            for x in 55..75 {
                mid += l(x, y);
            }
            for x in MID - 12..MID - 6 {
                gutter += l(x, y) * 20 / 6;
            }
        }
        assert!(mid > gutter, "the gutter should be shaded");
    }

    #[test]
    fn kinds_are_checked() {
        assert!(super::super::build(&Options { kind: "nope".into(), ..Options::default() }).is_err());
        assert!(super::super::build(&Options { kind: "book".into(), ..Options::default() }).is_ok());
    }
}
