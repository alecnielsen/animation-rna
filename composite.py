"""Batch composite all animation frames.

Takes pass1/pass2 frame pairs from renders/frames/ and composites them
using the same pipeline as render.py (translucent surface + edge outline).

Output: renders/composited/frame_NNNN.png

Run with: python3.11 composite.py [--debug]
  --debug: process only frames matching debug frame count (24)
"""

import numpy as np
from PIL import Image, ImageFilter
import os
import sys
import glob

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = "--debug" in sys.argv
FRAMES_DIR = "renders/frames"
OUTPUT_DIR = "renders/composited"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Compositing parameters (same as render.py)
OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 3
SURFACE_OPACITY = 0.20


def composite_frame(pass1_path, pass2_path, output_path):
    """Composite a single frame from pass1 (internal) and pass2 (surface).

    Same pipeline as render.py:
    1. Layer 1: Translucent surface overlay on internal components
    2. Layer 2: Edge outline from surface alpha channel
    """
    atoms = Image.open(pass1_path).convert("RGBA")
    surface = Image.open(pass2_path).convert("RGBA")

    # Layer 1: Translucent surface overlay
    surface_np = np.array(surface).astype(np.float32)
    surface_np[:, :, 3] = SURFACE_OPACITY * 255
    translucent = Image.fromarray(surface_np.astype(np.uint8), "RGBA")
    result = Image.alpha_composite(atoms, translucent)

    # Layer 2: Outer silhouette from alpha channel
    alpha = np.array(surface)[:, :, 3]
    alpha_mask = (alpha > 10).astype(np.uint8) * 255
    mask_img = Image.fromarray(alpha_mask)
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
    mask_img = Image.fromarray((np.array(mask_img) > 128).astype(np.uint8) * 255)
    silhouette = mask_img.filter(ImageFilter.FIND_EDGES)
    sil_np = (np.array(silhouette) > 30).astype(np.uint8) * 255
    sil_img = Image.fromarray(sil_np)
    for _ in range(OUTLINE_THICKNESS // 2):
        sil_img = sil_img.filter(ImageFilter.MaxFilter(3))

    edges_np = np.array(sil_img)
    overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
    mask = edges_np > 100
    overlay[mask, 0] = OUTLINE_COLOR[0]
    overlay[mask, 1] = OUTLINE_COLOR[1]
    overlay[mask, 2] = OUTLINE_COLOR[2]
    overlay[mask, 3] = 255

    result = Image.alpha_composite(result, Image.fromarray(overlay, "RGBA"))
    result.save(output_path)


def main():
    # Find all pass1 frames
    pass1_files = sorted(glob.glob(os.path.join(FRAMES_DIR, "pass1_*.png")))
    if not pass1_files:
        print(f"No pass1 frames found in {FRAMES_DIR}/")
        return

    print(f"=== Compositing {len(pass1_files)} frames ===")

    for pass1_path in pass1_files:
        # Extract frame number
        basename = os.path.basename(pass1_path)
        frame_num = basename.replace("pass1_", "").replace(".png", "")

        pass2_path = os.path.join(FRAMES_DIR, f"pass2_{frame_num}.png")
        if not os.path.exists(pass2_path):
            print(f"  SKIP frame {frame_num}: missing pass2")
            continue

        output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_num}.png")
        composite_frame(pass1_path, pass2_path, output_path)
        print(f"  Composited frame {frame_num}")

    print(f"\n=== Done! Composited frames in {OUTPUT_DIR}/ ===")
    print("Next: python3.11 encode.py [--debug]")


if __name__ == "__main__":
    main()
