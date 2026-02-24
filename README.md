# animation-rna

Molecular visualization of protein translation, rendered with Blender and [Molecular Nodes](https://github.com/BradyAJohnston/MolecularNodes).

Shows a timecourse of amino acid incorporation with real PDB structures: mRNA threaded through a ribosome, tRNA delivery, peptide bond formation, and a growing nascent polypeptide chain.

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
python3.11 build_tunnel_polypeptide.py   # ~15-30 min (100K MD steps)
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

# 0a. Build extended mRNA (tiles chain A4 x10, 500K-step 3-stage annealing + wall repulsion)
#     Writes extended_mrna.pdb (~40-60 min on CPU)
python3.11 build_extended_mrna.py

# 0b. Build tunnel-threaded polypeptide (traces exit tunnel, extended backbone in tunnel,
#     geometric de-clash, 100K-step 3-stage MD with wall repulsion)
#     Writes tunnel_polypeptide.pdb (~15-30 min)
python3.11 build_tunnel_polypeptide.py

# 0c. Compute PCA structural modes from MD trajectories
#     Writes mrna_modes.npz, trna_modes.npz (~30-60 min)
python3.11 compute_md_modes.py

# 1. Render all frames (single-pass, 10 elongation cycles)
python3.11 animate.py          # 1920x1080, 2400 frames (production)
python3.11 animate.py --debug  # 480x270, 240 frames (fast preview)

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
| `build_extended_mrna.py` | `extended_mrna.pdb` | Tile chain A4 x10, randomize sequence, 500K-step 3-stage MD annealing with ribosome wall repulsion |
| `build_tunnel_polypeptide.py` | `tunnel_polypeptide.pdb` | Trace exit tunnel, extended backbone in tunnel, helix after exit, geometric de-clash, 100K-step 3-stage MD with wall repulsion |
| `render_single_frame.py` | `renders/single_frame.png` | Single-frame render of full translation complex (ribosome, mRNA, tRNAs, polypeptide) |
| `compute_md_modes.py` | `mrna_modes.npz`, `trna_modes.npz` | PCA modes from MD for structural deformation |
| `build_extended_polypeptide.py` | `extended_polypeptide.pdb` | (legacy) Simple ideal helix aligned to C4 |

## Visual style spec

See [STYLE.md](STYLE.md) for the full visual design specification.

## Animation plan

See [PLAN.md](PLAN.md) for the full animation plan and technical approach.
