from pathlib import Path

from PIL import Image

RNG_SEED = 42


def render_plot(plot, tmp_path: Path) -> Image.Image:
    out = tmp_path / "_render.png"
    plot.save(str(out))
    img = Image.open(out).copy()
    out.unlink()
    return img
