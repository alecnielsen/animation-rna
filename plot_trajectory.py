"""Plot choreography trajectories for one elongation cycle.

Validates that mRNA, tRNAs, and polypeptide move correctly across frames
without needing to run a Blender render. Shows position over time for each
molecule, phase boundaries, and loop closure.

Run with: python3.11 plot_trajectory.py
Output: renders/trajectory_plot.png
"""

import numpy as np
import math
import sys
import os

# We need the choreography functions from animate.py but can't import
# directly (it imports bpy at module level). Instead, replicate the
# minimal logic here.

# ---------------------------------------------------------------------------
# Configuration (mirrored from animate.py)
# ---------------------------------------------------------------------------
FRAMES_PER_CYCLE = 240  # always plot at full resolution

PA_VEC = np.array((-2.51, 1.86, 0.05))
EP_VEC = -PA_VEC
CODON_SHIFT = np.array((-0.75, 0.35, -0.56))
ENTRY_OFFSET = 3.0 * PA_VEC
DEPART_OFFSET = 3.0 * EP_VEC


def scale_frames(f):
    return int(round(f * FRAMES_PER_CYCLE / 240))


def frame_t(frame, start, end):
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / (end - start)


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


# ---------------------------------------------------------------------------
# v4 schedule positions
# ---------------------------------------------------------------------------
def get_positions(local_frame):
    f0 = scale_frames(0)
    f12 = scale_frames(12)
    f96 = scale_frames(96)
    f120 = scale_frames(120)
    f144 = scale_frames(144)
    f192 = scale_frames(192)
    f240 = scale_frames(240)

    zero = np.zeros(3)

    # mRNA
    if local_frame < f144:
        mrna_delta = zero.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        mrna_delta = lerp(zero, CODON_SHIFT, t)
    else:
        mrna_delta = CODON_SHIFT.copy()

    # P-site tRNA
    if local_frame < f144:
        trna_p_delta = zero.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        trna_p_delta = lerp(zero, EP_VEC, t)
    elif local_frame < f240:
        t = frame_t(local_frame, f192, f240)
        trna_p_delta = lerp(EP_VEC, EP_VEC + DEPART_OFFSET, t)
    else:
        trna_p_delta = EP_VEC + DEPART_OFFSET

    # A-site tRNA
    if local_frame < f12:
        trna_a_delta = PA_VEC + ENTRY_OFFSET
    elif local_frame < f96:
        t = frame_t(local_frame, f12, f96)
        trna_a_delta = lerp(PA_VEC + ENTRY_OFFSET, PA_VEC, t)
    elif local_frame < f144:
        trna_a_delta = PA_VEC.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        trna_a_delta = lerp(PA_VEC, zero, t)
    else:
        trna_a_delta = zero.copy()

    return mrna_delta, trna_p_delta, trna_a_delta


def get_tumble_factor(local_frame, site):
    RESIDUAL_TUMBLE = 0.05
    f12 = scale_frames(12)
    f96 = scale_frames(96)
    f120 = scale_frames(120)
    f192 = scale_frames(192)
    f240 = scale_frames(240)

    if site == "A":
        if local_frame < f12:
            return 1.0
        elif local_frame < f96:
            return 1.0
        elif local_frame < f120:
            t = frame_t(local_frame, f96, f120)
            return max(RESIDUAL_TUMBLE, 1.0 - t)
        else:
            return RESIDUAL_TUMBLE
    elif site == "P":
        if local_frame < f192:
            return RESIDUAL_TUMBLE
        elif local_frame < f240:
            t = frame_t(local_frame, f192, f240)
            return RESIDUAL_TUMBLE + (1.0 - RESIDUAL_TUMBLE) * t
        else:
            return 1.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("ERROR: matplotlib required. Install with: pip install matplotlib")
        return

    frames = list(range(FRAMES_PER_CYCLE))
    n = len(frames)

    # Collect trajectories
    mrna_pos = np.zeros((n, 3))
    trna_p_pos = np.zeros((n, 3))
    trna_a_pos = np.zeros((n, 3))
    tumble_a = np.zeros(n)
    tumble_p = np.zeros(n)

    for i, f in enumerate(frames):
        m, tp, ta = get_positions(f)
        mrna_pos[i] = m
        trna_p_pos[i] = tp
        trna_a_pos[i] = ta
        tumble_a[i] = get_tumble_factor(f, "A")
        tumble_p[i] = get_tumble_factor(f, "P")

    # Phase boundaries (in 240-frame schedule)
    phases = [
        (0, 12, "ESTABLISH", "#444444"),
        (12, 96, "DELIVERY", "#2196F3"),
        (96, 120, "ACCOM.", "#4CAF50"),
        (120, 144, "PEPTIDE", "#FF9800"),
        (144, 192, "TRANSLOC.", "#9C27B0"),
        (192, 240, "DEPARTURE", "#F44336"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("v4 Choreography — One Elongation Cycle (240 frames)", fontsize=14)

    # Add phase background to all axes
    for ax in axes:
        for start, end, label, color in phases:
            ax.axvspan(start, end, alpha=0.08, color=color)
            if ax == axes[0]:
                mid = (start + end) / 2
                ax.text(mid, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                        label, ha='center', va='bottom', fontsize=7,
                        color=color, fontweight='bold')

    # Plot 1: mRNA displacement (magnitude)
    ax = axes[0]
    mrna_mag = np.linalg.norm(mrna_pos, axis=1)
    ax.plot(frames, mrna_pos[:, 0], 'b-', alpha=0.6, label='X')
    ax.plot(frames, mrna_pos[:, 1], 'g-', alpha=0.6, label='Y')
    ax.plot(frames, mrna_pos[:, 2], 'r-', alpha=0.6, label='Z')
    ax.plot(frames, mrna_mag, 'k-', linewidth=2, label='|total|')
    ax.set_ylabel("mRNA delta (BU)")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title("mRNA Translation (one codon shift per cycle)")
    # Re-add phase labels after ylim is set
    for start, end, label, color in phases:
        ax.axvspan(start, end, alpha=0.08, color=color)

    # Plot 2: P-site tRNA displacement
    ax = axes[1]
    trna_p_mag = np.linalg.norm(trna_p_pos, axis=1)
    ax.plot(frames, trna_p_pos[:, 0], 'b-', alpha=0.6, label='X')
    ax.plot(frames, trna_p_pos[:, 1], 'g-', alpha=0.6, label='Y')
    ax.plot(frames, trna_p_pos[:, 2], 'r-', alpha=0.6, label='Z')
    ax.plot(frames, trna_p_mag, 'k-', linewidth=2, label='|total|')
    ax.set_ylabel("P-tRNA delta (BU)")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title("P-site tRNA (bound → translocate → depart)")
    for start, end, label, color in phases:
        ax.axvspan(start, end, alpha=0.08, color=color)

    # Plot 3: A-site tRNA displacement
    ax = axes[2]
    trna_a_mag = np.linalg.norm(trna_a_pos, axis=1)
    ax.plot(frames, trna_a_pos[:, 0], 'b-', alpha=0.6, label='X')
    ax.plot(frames, trna_a_pos[:, 1], 'g-', alpha=0.6, label='Y')
    ax.plot(frames, trna_a_pos[:, 2], 'r-', alpha=0.6, label='Z')
    ax.plot(frames, trna_a_mag, 'k-', linewidth=2, label='|total|')
    ax.set_ylabel("A-tRNA delta (BU)")
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title("A-site tRNA (approach → accommodate → translocate to P)")
    for start, end, label, color in phases:
        ax.axvspan(start, end, alpha=0.08, color=color)

    # Plot 4: Tumble factors
    ax = axes[3]
    ax.plot(frames, tumble_a, 'orange', linewidth=2, label='A-site tumble')
    ax.plot(frames, tumble_p, 'purple', linewidth=2, label='P-site tumble')
    ax.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5, label='residual (5%)')
    ax.set_ylabel("Tumble factor")
    ax.set_xlabel("Frame (within cycle)")
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title("tRNA Tumble Amplitude")
    ax.set_ylim(-0.05, 1.1)
    for start, end, label, color in phases:
        ax.axvspan(start, end, alpha=0.08, color=color)

    # Add phase labels to top plot
    ax0_ylim = axes[0].get_ylim()
    for start, end, label, color in phases:
        mid = (start + end) / 2
        axes[0].text(mid, ax0_ylim[1] * 0.95, label, ha='center', va='top',
                     fontsize=7, color=color, fontweight='bold')

    # Loop closure check
    mrna_0 = mrna_pos[0]
    mrna_end = mrna_pos[-1] - CODON_SHIFT  # after subtracting one cycle's shift
    trna_a_0 = trna_a_pos[0]
    trna_a_end = trna_a_pos[-1]
    trna_p_0 = trna_p_pos[0]

    fig.text(0.02, 0.02,
             f"Loop closure: mRNA Δ={np.linalg.norm(mrna_end - mrna_0):.4f} BU | "
             f"A-tRNA start={np.linalg.norm(trna_a_0):.2f}, "
             f"A-tRNA end→P-site={np.linalg.norm(trna_a_end):.4f} BU | "
             f"P-tRNA start={np.linalg.norm(trna_p_0):.4f} BU (should≈0)",
             fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    os.makedirs("renders", exist_ok=True)
    out = "renders/trajectory_plot.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.close()

    # Print summary
    print(f"\n=== Trajectory Summary ===")
    print(f"  mRNA: starts at origin, ends at CODON_SHIFT = {CODON_SHIFT}")
    print(f"  A-tRNA: starts at {trna_a_pos[0]} (far), ends at {trna_a_pos[-1]} (P-site)")
    print(f"  P-tRNA: starts at {trna_p_pos[0]} (P-site), ends at {trna_p_pos[-1]} (departed)")
    print(f"  Loop closure mRNA: {np.linalg.norm(mrna_end - mrna_0):.6f} BU")
    print(f"  A-tRNA end = P-site? |delta| = {np.linalg.norm(trna_a_end):.6f} BU")


if __name__ == "__main__":
    main()
