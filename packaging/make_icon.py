"""Write packaging/seisdesk.ico (wiggle traces on a blue tile) with NumPy and the standard library only."""

import os
import struct
import zlib

import numpy as np


def render(size: int) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) + 0.5
    s = size / 256.0
    img = np.zeros((size, size, 4), dtype=np.float64)
    r = 40 * s                                                  # rounded tile
    dx = np.maximum(np.maximum(r - x, x - (size - r)), 0)
    dy = np.maximum(np.maximum(r - y, y - (size - r)), 0)
    inside = np.clip(r + 0.5 - np.hypot(dx, dy), 0, 1)
    img[..., :3] = np.array([31, 78, 121]) / 255.0
    img[..., 3] = inside
    t = y / size
    for k, centre in enumerate((0.28, 0.5, 0.72)):
        arg = (np.pi * 9.0 * (t - 0.38 - 0.1 * k)) ** 2
        wav = (1 - 2 * arg) * np.exp(-arg) + 0.35 * np.sin(2 * np.pi * 7 * t + k)
        xc = centre * size + 22 * s * wav
        base = centre * size
        fill = (x >= base) & (x <= xc)
        line = np.clip(1.6 * s + 0.5 - np.abs(x - xc), 0, 1)
        white = np.maximum(fill.astype(float), line) * inside
        img[..., :3] = img[..., :3] * (1 - white[..., None]) + white[..., None]
    return (np.clip(img, 0, 1) * 255).round().astype(np.uint8)


def png(rgba: np.ndarray) -> bytes:
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[i].tobytes() for i in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main() -> None:
    sizes = (256, 64, 48, 32, 16)
    images = [png(render(n)) for n in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = b""
    for n, data in zip(sizes, images):
        entries += struct.pack("<BBBBHHII", n % 256, n % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seisdesk.ico")
    with open(out, "wb") as f:
        f.write(header + entries + b"".join(images))
    print("wrote", out)


if __name__ == "__main__":
    main()
