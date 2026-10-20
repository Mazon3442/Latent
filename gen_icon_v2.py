"""Generate a 32x32 .ico with BITMAPV4HEADER (modern format) for Tauri."""
import struct
import sys
import os

icon_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app", "src-tauri", "icons", "icon.ico"
)
w, h = 32, 32

# ICONDIR header (6 bytes)
header = struct.pack("<HHH", 0, 1, 1)

# BITMAPV4HEADER (108 bytes) – sRGB
v4_size = 108
red_mask   = 0x00FF0000
green_mask = 0x0000FF00
blue_mask  = 0x000000FF
alpha_mask = 0xFF000000
cs_type    = "sRGB"  # fourcc
endpoints  = struct.pack("<9I",
    0, 0, 0,          # R
    100000, 0, 0,     # G
    0, 0, 100000      # B
)
gamma = 1000000  # 2.2 in 1/1000000 units
bit_count = 32

# Pixel data: bottom-up ARGB rows, padded to 4-byte boundary
bytes_per_pixel = bit_count // 8
raw_row = w * bytes_per_pixel
row_padded = ((raw_row + 3) // 4) * 4
pixel_data = b"\x00" * (row_padded * h)

# File size = header + entry + BITMAPV4HEADER + pixel_data
bmp_header_size = v4_size + len(pixel_data)
file_data_offset = 6 + 16
file_size = file_data_offset + bmp_header_size

# BITMAPV4HEADER (108 bytes)
v4 = struct.pack("<I", v4_size)
v4 += struct.pack("<III", red_mask, green_mask, blue_mask)
v4 += struct.pack("<I", alpha_mask)
v4 += struct.pack("<I", 0x73524742)  # sRGB fourcc
v4 += struct.pack("<9I", 0, 0, 0, 100000, 0, 0, 0, 0, 100000)
v4 += struct.pack("<III", gamma, gamma, gamma)
v4 += struct.pack("<II", 0, 0)  # render/clip (reserved)

# Now pack the pixel data: ARGB format
# For a black icon, all pixels are 0x00000000 (BGRA with alpha=0)
# We need to create a visible icon, so let's make a simple gradient
pixel_data = bytearray(row_padded * h)
for y in range(h):
    row_start = y * row_padded
    for x in range(w):
        off = row_start + (x * 4)
        # Gradient from dark to accent color
        t = x / max(w - 1, 1)
        a = 255
        r = int(17 * (1 - t) + 255 * t)
        g = int(18 * (1 - t) + 170 * t)
        b = int(19 * (1 - t) + 44 * t)
        pixel_data[off]     = b  # B
        pixel_data[off + 1] = g  # G
        pixel_data[off + 2] = r  # R
        pixel_data[off + 3] = a  # A
pixel_data = bytes(pixel_data)

# Recheck sizes
bmp_data = v4 + pixel_data
bmp_size = len(bmp_data)
file_size = file_data_offset + bmp_size

# ICONDIRENTRY (16 bytes)
entry = struct.pack(
    "<BBBBHHII",
    w, h,               # width, height (0 = 256 if < 256)
    0,                  # color count
    0,                  # reserved
    1,                  # color planes
    bit_count,          # bits per pixel
    bmp_size,           # bytes in resource
    file_data_offset,   # offset to resource
)

with open(icon_path, "wb") as f:
    f.write(header + entry + bmp_data)

print(f"Created {icon_path} ({bmp_size} BMP data, {file_size} total)")
