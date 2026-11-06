"""Generate the two material maps and the shared black fallback. Needs Pillow and NumPy."""
from pathlib import Path

import numpy as np
from PIL import Image

TEXTURES = Path(__file__).resolve().parents[1] / "textures"


def perlin(width, height, cells, seed):
    """Periodic 2D gradient noise with quintic interpolation."""
    angles = np.random.default_rng(seed).uniform(0, 2 * np.pi, (cells, cells))
    gx, gy = np.cos(angles), np.sin(angles)
    x, y = np.meshgrid(np.arange(width) * cells / width, np.arange(height) * cells / height)
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - ix, y - iy

    def dot(dx, dy):
        return (gx[(iy + dy) % cells, (ix + dx) % cells] * (fx - dx) +
                gy[(iy + dy) % cells, (ix + dx) % cells] * (fy - dy))

    u, v = fx**3 * (fx * (fx * 6 - 15) + 10), fy**3 * (fy * (fy * 6 - 15) + 10)
    top = dot(0, 0) * (1 - u) + dot(1, 0) * u
    bottom = dot(0, 1) * (1 - u) + dot(1, 1) * u
    return top * (1 - v) + bottom * v


def generate_maps():
    with Image.open(TEXTURES / "water.png") as image:
        water = np.asarray(image.convert("RGB"), dtype=float) / 255
    luminance = water @ np.array([0.2126, 0.7152, 0.0722])
    low, high = np.quantile(luminance, [0.15, 0.85])
    specular = np.clip((luminance - low) / max(high - low, 0.0001), 0, 1)
    Image.fromarray(np.uint8(np.rint(specular * 255))).save(TEXTURES / "water_specular.png")

    with Image.open(TEXTURES / "night-stars.png") as image:
        stars = np.asarray(image.convert("RGB"), dtype=float) / 255
    height, width = stars.shape[:2]
    noise = sum(weight * perlin(width, height, cells, 37 + cells)
                for cells, weight in [(6, 0.6), (12, 0.3), (24, 0.1)])
    mask = np.clip((noise + 0.18) / 0.45, 0, 1)
    # Leave dark texels black and light up bright details in Perlin-shaped regions.
    brightness = stars.max(axis=2)
    detail = np.clip((brightness - 0.12) / 0.45, 0, 1)
    emissive = np.clip(stars * (mask * detail)[..., None], 0, 1)
    Image.fromarray(np.uint8(np.rint(emissive * 255))).save(TEXTURES / "night-stars_emissive.png")
    Image.new("RGBA", (1, 1), (0, 0, 0, 255)).save(TEXTURES / "black.png")


if __name__ == "__main__":
    generate_maps()
