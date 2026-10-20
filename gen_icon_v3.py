"""Generate a 32x32 .ico with full BITMAPV4HEADER (modern format) for Tauri.
BITMAPV4HEADER = BITMAPINFOHEADER(40) + V4 extension(68) = 108 bytes."""
import struct
import os

icon_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app", "src-tauri", "icons", "icon.ico"
)
W, H = 32, 32
BPP = 32

# ---- ICONDIR (6 bytes) ----
icondir = struct.pack("<HHH", 0, 1, 1)  # reserved=0, type=1(icon), count=1

# ---- BITMAPINFOHEADER (40 bytes) ----
bi_size = 108          # biSize = size of entire BITMAPV4HEADER structure
bi_width = W
bi_height = H          # single-direction height (V4 format)
bi_planes = 1
bi_bitcount = BPP
bi_compression = 0     # BI_RGB
bi_sizeimage = 0
bi_xpelspermeter = 0
bi_ypelspermeter = 0
bi_clrused = 0
bi_clrimp = 0

bih = struct.pack("<I", bi_size)
bih += struct.pack("<i", bi_width)
bih += struct.pack("<i", bi_height)
bih += struct.pack("<HH", bi_planes, bi_bitcount)
bih += struct.pack("<III", bi_compression, bi_sizeimage, 0)
bih += struct.pack("<II", 0, bi_clrused)
bih += struct.pack("<I", bi_clrimp)
# bih = 40 bytes

# ---- V4 extension (68 bytes) ----
v4_ext = struct.pack("<I", 68)                  # biV4Size (size of V4 extension)
v4_ext += struct.pack("<III", 0x00FF0000, 0x0000FF00, 0x000000FF)  # masks
v4_ext += struct.pack("<I", 0xFF000000)         # alpha mask
v4_ext += struct.pack("<I", 0x73524742)         # sRGB fourcc
# CIEXTRIPLE: 3 endpoints × 3 components × LONG
v4_ext += struct.pack("<9I",
    0, 0, 0,            # Red endpoint
    100000, 0, 0,       # Green endpoint
    0, 0, 100000)       # Blue endpoint
v4_ext += struct.pack("<III", 1000000, 1000000, 1000000)  # gamma
v4_ext += struct.pack("<II", 0, 0)              # intent, compression
v4_ext += struct.pack("<I", 0)                  # BTC index
v4_ext += struct.pack("<QQ", 0, 0)              # reserved (2 × LARGE_INTEGER)
# v4_ext = 68 bytes

# ---- Pixel data (BMP row padding: rows are 4-byte aligned) ----
bytes_per_pixel = BPP // 8
raw_row = W * bytes_per_pixel
row_padded = ((raw_row + 3) // 4) * 4
pixel_buf = bytearray(row_padded * H)

for y in range(H):
    row_start = y * row_padded
    for x in range(W):
        off = row_start + x * 4
        t = x / max(W - 1, 1)
        a = 255
        r = int(17 * (1 - t) + 255 * t)
        g = int(18 * (1 - t) + 170 * t)
        b = int(19 * (1 - t) + 44 * t)
        pixel_buf[off]     = b
        pixel_buf[off + 1] = g
        pixel_buf[off + 2] = r
        pixel_buf[off + 3] = a

bmp_header = bih + v4_ext   # 40 + 68 = 108 bytes
bmp_data = bmp_header + bytes(pixel_buf)

# ---- ICONDIRENTRY (16 bytes) ----
entry = struct.pack(
    "<BBBBHHII",
    W, H,             # width, height (0 = 256)
    0,                # color count
    0,                # reserved
    1,                # color planes
    BPP,              # bits per pixel
    len(bmp_data),    # bytes in resource
    6 + 16,           # offset to resource (ICONDIR + ICONDIRENTRY)
)

with open(icon_path, "wb") as f:
    f.write(icondir + entry + bmp_data)

print(f"Created {icon_path}  ({len(bmp_data)} bytes BMP, {6+16+len(bmp_data)} total)")
