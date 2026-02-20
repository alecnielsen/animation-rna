"""Encode composited frames to video.

Reads renders/composited/frame_NNNN.png and produces:
- renders/ribosome_animation.mp4  (H.264, web-compatible)
- renders/ribosome_animation.webm (VP9, smaller file)

Run with: python3.11 encode.py [--debug]
  --debug: lower bitrate, same output names
"""

import subprocess
import os
import sys
import glob

DEBUG = "--debug" in sys.argv
INPUT_DIR = "renders/composited"
OUTPUT_DIR = "renders"

FPS = 24
CRF_H264 = 18 if not DEBUG else 28
CRF_VP9 = 30 if not DEBUG else 40


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def count_frames():
    frames = sorted(glob.glob(os.path.join(INPUT_DIR, "frame_*.png")))
    return len(frames)


def encode_h264(output_path):
    """Encode to MP4 with H.264 codec."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(INPUT_DIR, "frame_%04d.png"),
        "-c:v", "libx264",
        "-crf", str(CRF_H264),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def encode_vp9(output_path):
    """Encode to WebM with VP9 codec."""
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(INPUT_DIR, "frame_%04d.png"),
        "-c:v", "libvpx-vp9",
        "-crf", str(CRF_VP9),
        "-b:v", "0",
        "-pix_fmt", "yuva420p",
        output_path,
    ]
    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
        return

    n_frames = count_frames()
    if n_frames == 0:
        print(f"No composited frames found in {INPUT_DIR}/")
        return

    print(f"=== Encoding {n_frames} frames @ {FPS}fps ===")

    mp4_path = os.path.join(OUTPUT_DIR, "ribosome_animation.mp4")
    print(f"\n--- H.264 → {mp4_path} ---")
    encode_h264(mp4_path)
    mp4_size = os.path.getsize(mp4_path) / (1024 * 1024)
    print(f"  Output: {mp4_path} ({mp4_size:.1f} MB)")

    webm_path = os.path.join(OUTPUT_DIR, "ribosome_animation.webm")
    print(f"\n--- VP9 → {webm_path} ---")
    encode_vp9(webm_path)
    webm_size = os.path.getsize(webm_path) / (1024 * 1024)
    print(f"  Output: {webm_path} ({webm_size:.1f} MB)")

    print(f"\n=== Done! ===")
    print(f"  MP4:  {mp4_path} ({mp4_size:.1f} MB)")
    print(f"  WebM: {webm_path} ({webm_size:.1f} MB)")


if __name__ == "__main__":
    main()
