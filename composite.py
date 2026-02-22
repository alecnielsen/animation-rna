"""Batch composite all animation frames.

Takes pass1/pass2 frame pairs from renders/frames/ and composites them
with translucent surface overlay (no outline — v3).

Output: renders/composited/frame_NNNN.png

Run with: python3.11 composite.py [--debug]
  --debug: process only frames matching debug frame count (24)
"""

import numpy as np
from PIL import Image
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

# Compositing parameters
SURFACE_OPACITY = 0.35


def composite_frame(pass1_path, pass2_path, output_path):
    """Composite a single frame from pass1 (internal) and pass2 (surface).

    Translucent surface overlay on internal components at 35% opacity.
    """
    atoms = Image.open(pass1_path).convert("RGBA")
    surface = Image.open(pass2_path).convert("RGBA")

    # Translucent surface overlay
    surface_np = np.array(surface).astype(np.float32)
    surface_np[:, :, 3] = SURFACE_OPACITY * 255
    translucent = Image.fromarray(surface_np.astype(np.uint8), "RGBA")
    result = Image.alpha_composite(atoms, translucent)

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
