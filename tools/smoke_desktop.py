"""Exercise the actual desktop widgets and render a synthetic, non-private preview."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk
import numpy as np
from PIL import ImageGrab
from full_pipline.colony_pipeline_app import ColonyPipelineApp, write_rgb_image


def main():
    root = tk.Tk()
    app = ColonyPipelineApp(root)
    root.update()
    output = Path("outputs/qa")
    output.mkdir(parents=True, exist_ok=True)
    box = (root.winfo_rootx(), root.winfo_rooty(), root.winfo_rootx() + root.winfo_width(), root.winfo_rooty() + root.winfo_height())
    ImageGrab.grab(window=root.winfo_id()).save(output / "desktop-empty.png")
    # Exercise rendering and Unicode file IO without publishing real samples.
    pixels = np.full((128, 128, 3), (15, 118, 110), dtype=np.uint8)
    write_rgb_image(output / "проверка.png", pixels)
    app.preview_rgb = pixels
    app._draw_preview_image()
    root.update()
    assert app.preview_photo is not None
    root.destroy()
    print("Desktop widget/render/Unicode IO smoke test OK")


if __name__ == "__main__":
    main()
