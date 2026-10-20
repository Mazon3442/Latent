"""Generate a minimal 16x16 .ico file for the Tauri app."""
import struct
import sys
import os

icon_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app", "src-tauri", "icons", "icon.ico"
)
width = 16
height = 16

# --- ICONDIR header (6 bytes) ---
header = struct.pack("<HHH", 0, 1, 1)  # reserved=0, type=1(icon), count=1

# --- BMP image data ---
# BITMAPINFOHEADER
# height = height*2 because ICO uses separate XOR (pixel) and AND (mask) rows
bi_height = height * 2
# Bytes per row, padded to 4-byte boundary
row_bytes = (width + 7) // 8  # 2 bytes for 16 pixels
row_padded = ((row_bytes + 3) // 4) * 4  # pad to 4-byte boundary
pixel_data = b"\x00" * (row_padded * height)   # black pixels
mask_data = b"\xFF" * (row_padded * height)     # white mask (visible)

bmp_header = (
    struct.pack("<2s", b"BM")           # signature
    + struct.pack("<I", 54 + len(pixel_data) + len(mask_data))  # file size
    + struct.pack("<HH", 0, 0)          # reserved
    + struct.pack("<I", 54)             # data offset
    + struct.pack("<I", 40)             # DIB header size
    + struct.pack("<i", width)          # width
    + struct.pack("<i", bi_height)      # height (double for XOR+AND)
    + struct.pack("<H", 1)              # color planes
    + struct.pack("<H", 1)              # bits per pixel
    + struct.pack("<I", 0)              # compression (none)
    + struct.pack("<I", len(pixel_data) + len(mask_data))  # image size
    + struct.pack("<i", 0)              # h resolution
    + struct.pack("<i", 0)              # v resolution
    + struct.pack("<I", 0)              # colors used
    + struct.pack("<I", 0)              # important colors
)
bmp_data = bmp_header + pixel_data + mask_data

# --- ICONDIRENTRY (16 bytes) ---
entry = struct.pack(
    "<BBBBHHII",
    width, height, 0, 0,   # width, height, color count, reserved
    1, 1,                  # color planes, bits per pixel
    len(bmp_data),         # bytes in resource
    6 + 16 + len(bmp_data) # offset to resource
)

with open(icon_path, "wb") as f:
    f.write(header + entry + bmp_data)

print(f"Created {icon_path}")
