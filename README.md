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

## Animation

Four-step pipeline: build extended structures → render frames → composite → encode video.

```bash
source mn_env/bin/activate

# 0. Build extended mRNA (tiles chain A4 x10, writes extended_mrna.pdb)
python3.11 build_extended_mrna.py

# 1. Render all frames (two passes per frame)
python3.11 animate.py          # 1920x1080, 240 frames (production)
python3.11 animate.py --debug  # 480x270, 24 frames (fast preview)

# 2. Composite pass1 + pass2 for each frame
python3.11 composite.py

# 3. Encode to video
python3.11 encode.py
```

Output:
- `renders/frames/` — raw pass1/pass2 PNGs per frame
- `renders/composited/` — final composited PNGs
- `renders/ribosome_animation.mp4` — H.264 video
- `renders/ribosome_animation.webm` — VP9 video

## Visual style spec

See [STYLE.md](STYLE.md) for the full visual design specification.

## Animation plan

See [PLAN.md](PLAN.md) for the full animation plan and technical approach.
