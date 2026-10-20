"""Generate a minimal 16x16 .ico with BITMAPINFOHEADER (biSize=40) — works with RC.EXE."""
import struct
import os

icon_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app", "src-tauri", "icons", "icon.ico"
)
W, H = 16, 16  # RC.EXE is picky; keep it small
BPP = 32

# ICONDIR header (6 bytes)
icondir = struct.pack("<HHH", 0, 1, 1)  # reserved=0, type=1(icon), count=1

# BITMAPINFOHEADER (40 bytes) — biSize MUST be 40 for RC.EXE compatibility
bi_size = 40
bi_width = W
bi_height = H  # positive = bottom-up pixel order
bi_planes = 1
bi_bitcount = BPP
bi_compression = 0  # BI_RGB (uncompressed)
bi_sizeimage = 0
bi_xpels = 0
bi_ypels = 0
bi_clrused = 0
bi_clrimp = 0

bih = struct.pack(
    "<IIIHHIIIIII",
    bi_size, bi_width, bi_height, bi_planes, bi_bitcount,
    bi_compression, bi_sizeimage, bi_xpels, bi_ypels,
    bi_clrused, bi_clrimp,
)
assert len(bih) == 40

# Pixel data: bottom-up rows, 4-byte aligned
bytes_per_pixel = BPP // 8  # 4
raw_row = W * bytes_per_pixel  # 64
row_padded = ((raw_row + 3) // 4) * 4  # 64 (already aligned)
pixel_buf = bytearray(row_padded * H)

# Draw a simple icon: dark bg with white circles and amber accent
for y in range(H):
    row_start = y * row_padded
    for x in range(W):
        off = row_start + x * 4
        # Background: dark (#111213 → BGRA: 13 12 11 FF)
        r, g, b, a = 0x11, 0x12, 0x13, 0xFF
        # Four circles (like our SVG mark)
        cx1, cy1 = 4, 4  # top-left
        cx2, cy2 = 11, 4  # top-right
        cx3, cy3 = 4, 11  # bottom-left
        cx4, cy4 = 11, 11  # bottom-right (accent)
        for cx, cy, accent in [(cx1, cy1, False), (cx2, cy2, False), (cx3, cy3, False), (cx4, cy4, True)]:
            dx, dy = x - cx, y - cy
            if dx * dx + dy * dy <= 4:  # radius ~2
                if accent:
                    r, g, b = 0xFF, 0xAA, 0x2C
                a = 0xFF
        # BMP stores BGRA
        pixel_buf[off]     = b
        pixel_buf[off + 1] = g
        pixel_buf[off + 2] = r
        pixel_buf[off + 3] = a

bmp_data = bih + bytes(pixel_buf)
bmp_size = len(bmp_data)
offset = 6 + 16  # ICONDIR + ICONDIRENTRY

# ICONDIRENTRY (16 bytes)
entry = struct.pack(
    "<BBBBHHII",
    W, H,             # width, height (0 = 256 if < 256)
    0,                # color count
    0,                # reserved
    1,                # color planes
    BPP,              # bits per pixel
    bmp_size,         # bytes in resource
    offset,           # offset to resource
)

with open(icon_path, "wb") as f:
    f.write(icondir + entry + bmp_data)

print(f"Created {icon_path}  ({bmp_size} BMP, {offset+bmp_size} total)")
