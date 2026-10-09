"""Erzeugt ``packaging/assets/nova.ico`` (16–256 px) aus dem NOVA-Logo (api/static/favicon.svg).

Reproduzierbar ohne SVG-Renderer: gleiche Geometrie (64er-Raster), mit Überabtastung gezeichnet.

    python packaging/make_icon.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "assets" / "nova.ico"
SIZES = [16, 20, 24, 32, 40, 48, 64, 128, 256]
# Pfad aus favicon.svg: M18 46V18h6l16 18V18h6v28h-6L24 28v18z (Raster 64×64)
N_SHAPE = [(18, 46), (18, 18), (24, 18), (40, 36), (40, 18), (46, 18), (46, 46), (40, 46),
           (24, 28), (24, 46)]  # fmt: skip
BG = (15, 18, 23, 255)  # #0f1217
GRAD = ((154, 166, 255), (108, 92, 231))  # #9aa6ff → #6c5ce7


def render(size: int = 256, scale: int = 4) -> Image.Image:
    big = size * scale
    unit = big / 64
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle((0, 0, big - 1, big - 1), radius=16 * unit, fill=BG)
    gradient = Image.new("RGBA", (big, big))
    pixels = gradient.load()
    assert pixels is not None
    for y in range(big):
        for x in range(big):
            t = (x + y) / (2 * (big - 1))
            r, g, b = (round(a + (b - a) * t) for a, b in zip(*GRAD, strict=True))
            pixels[x, y] = (r, g, b, 255)
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).polygon([(x * unit, y * unit) for x, y in N_SHAPE], fill=255)
    img.paste(gradient, (0, 0), mask)
    return img.resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    render(256).save(OUT, format="ICO", sizes=[(s, s) for s in SIZES])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
