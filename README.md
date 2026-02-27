# animation-rna

Molecular visualization of protein translation, rendered with Blender and [Molecular Nodes](https://github.com/BradyAJohnston/MolecularNodes).

Shows a seamless-looping animation of amino acid incorporation with real PDB structures: mRNA threaded through a ribosome, tRNA delivery/accommodation/departure choreography, and a nascent polypeptide chain of repeating Villin HP35 folded domains scrolling away from the ribosome exit tunnel.

## Setup

Requires Python 3.11 (strict Blender requirement).

```bash
brew install python@3.11

python3.11 -m venv mn_env
source mn_env/bin/activate
pip install "molecularnodes[bpy]"
```

## Rendering

Scripts run headlessly via the `bpy` Python module — no Blender GUI needed.

```bash
source mn_env/bin/activate

# Style development (small test structure, fast iteration)
python3.11 test_render.py

# Single-frame ribosome render (slow, only after style is locked)
python3.11 render.py
```

Output goes to `renders/`.

## Single-frame render

Quick validation of all molecule placements before running the full animation.

```bash
source mn_env/bin/activate

# Build structures first (see Preprocessing below)
python3.11 build_tunnel_polypeptide.py   # Repeating HP35 domains (~5 min)
python3.11 build_extended_mrna.py        # ~40-60 min (500K MD steps)

# Render single frame
python3.11 render_single_frame.py --debug  # 960x540, 32 samples (~2-5 min)
python3.11 render_single_frame.py          # 1920x1080, 128 samples (~10-20 min)
```

Output: `renders/single_frame.png`

## Animation

Multi-step pipeline: build extended structures → compute modes → render frames → encode video.

```bash
source mn_env/bin/activate

# 0a. Build extended mRNA (tiles chain A4 x20, 500K-step 3-stage annealing + wall repulsion,
#     shifts to decoding center for tRNA-mRNA base pairing)
#     Writes extended_mrna.pdb (~40-60 min on CPU)
python3.11 build_extended_mrna.py

# 0b. Build repeating polypeptide (traces exit tunnel, places 8 Villin HP35 domains
#     with GSG linkers, outputs folded PDB + dual-coordinate NPZ for morph animation)
#     Writes repeating_polypeptide.pdb, repeating_polypeptide_folds.npz (~5 min)
python3.11 build_tunnel_polypeptide.py

# 0c. Compute PCA structural modes from MD trajectories
#     Writes mrna_modes.npz, trna_modes.npz (~30-60 min)
python3.11 compute_md_modes.py

# 1. Render all frames (38 elongation cycles for seamless loop)
python3.11 animate.py          # 1920x1080, 456 frames (production, 19s @ 24fps)
python3.11 animate.py --debug  # 480x270, 228 frames (fast preview, 9.5s @ 24fps)

# 2. Encode to video
python3.11 encode.py
python3.11 encode.py --debug
```

Output:
- `renders/frames/` — rendered PNGs per frame
- `renders/ribosome_animation.mp4` — H.264 video
- `renders/ribosome_animation.webm` — VP9 video

## Preprocessing scripts

| Script | Output | Purpose |
|--------|--------|---------|
| `build_extended_mrna.py` | `extended_mrna.pdb` | Tile chain A4 x20, 500K-step 3-stage MD annealing with ribosome wall repulsion, shift to decoding center |
| `build_tunnel_polypeptide.py` | `repeating_polypeptide.pdb`, `repeating_polypeptide_folds.npz` | Trace exit tunnel, place 8 repeating Villin HP35 (1YRF) folded domains with GSG linkers. NPZ stores dual extended/folded coordinates per domain for morph animation |
| `render_single_frame.py` | `renders/single_frame.png` | Single-frame render of full translation complex (ribosome, mRNA, tRNAs, polypeptide) |
| `compute_md_modes.py` | `mrna_modes.npz`, `trna_modes.npz` | PCA modes from MD for structural deformation |
| `animate.py` | `renders/frames/` | 38-cycle seamless loop: 6-phase tRNA choreography, polypeptide folding morph (extended→folded with N-to-C wave), domain scrolling |
| `encode.py` | `renders/ribosome_animation.mp4`, `.webm` | Encode rendered frames to H.264 and VP9 video |

## Visual style spec

See [STYLE.md](STYLE.md) for the full visual design specification.

## Animation plan

See [PLAN.md](PLAN.md) for the full animation plan and technical approach.
