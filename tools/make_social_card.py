"""Generate the Open Graph card from the same procedural body as the assay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from lentoorava.pulse_repair import apply_wounds, make_symmetric_body


WIDTH, HEIGHT = 1200, 630
BG = (6, 16, 20)
CYAN = (102, 251, 209)
ACID = (199, 255, 87)
MUTED = (139, 166, 158)
PINK = (255, 84, 127)


def font(path: str, size: int):
    return ImageFont.truetype(path, size)


SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def main():
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas, "RGBA")

    for x in range(0, WIDTH, 42):
        draw.line((x, 0, x, HEIGHT), fill=(120, 180, 165, 12), width=1)
    for y in range(0, HEIGHT, 42):
        draw.line((0, y, WIDTH, y), fill=(120, 180, 165, 12), width=1)

    glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow, "RGBA")
    glow_draw.ellipse((690, -90, 1320, 650), fill=(40, 210, 170, 42))
    glow = glow.filter(ImageFilter.GaussianBlur(100))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), glow)
    draw = ImageDraw.Draw(canvas, "RGBA")

    draw.text((62, 54), "LENTOORAVA / PULSE REPAIR", font=font(MONO, 18), fill=CYAN)
    draw.text((62, 119), "CAN A BODY", font=font(SANS_BOLD, 66), fill=(233, 248, 242))
    draw.text((62, 191), "FIND A WOUND", font=font(SANS_BOLD, 66), fill=(233, 248, 242))
    draw.text((62, 263), "IT CANNOT SEE?", font=font(SANS_BOLD, 66), fill=ACID)
    draw.text(
        (65, 367),
        "BOUNDED LOCAL WORKERS\nONE GLOBAL NUMBER\nNO BACKPROP",
        font=font(MONO, 21),
        spacing=13,
        fill=MUTED,
    )

    target = make_symmetric_body(96)
    damaged = apply_wounds(target, [(0.62, -0.04, 0.18), (0.76, 0.18, 0.11)])
    image = Image.fromarray(np.uint8(np.clip(damaged, 0, 1) * 255), mode="RGB")
    image = image.resize((520, 520), Image.Resampling.BICUBIC)
    body_glow = image.filter(ImageFilter.GaussianBlur(18))
    body_glow.putalpha(70)
    canvas.alpha_composite(body_glow, (660, 56))
    canvas.alpha_composite(image.convert("RGBA"), (660, 56))

    draw = ImageDraw.Draw(canvas, "RGBA")
    draw.line((920, 72, 920, 572), fill=(*ACID, 105), width=1)
    boxes = [(955, 224, 1130, 420), (1012, 278, 1128, 420), (1056, 312, 1128, 420)]
    for i, box in enumerate(boxes):
        draw.rectangle(box, outline=(*CYAN, 90 + i * 45), width=2)
    draw.ellipse((1077, 348, 1094, 365), fill=ACID)
    draw.arc((780, 226, 1100, 474), start=200, end=340, fill=(*CYAN, 160), width=3)

    draw.rounded_rectangle((61, 525, 606, 583), radius=6, fill=(10, 28, 30, 220), outline=(120, 210, 185, 48), width=1)
    draw.text((82, 542), "36 PULSES  →  81.1% DAMAGE RECOVERED", font=font(MONO, 17), fill=(210, 231, 224))
    draw.text((866, 591), "anttiluode.github.io/LentoOrava", font=font(MONO, 14), fill=MUTED)

    out = Path("assets/pulse-repair-card.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out, quality=94, optimize=True)
    print(out)


if __name__ == "__main__":
    main()

