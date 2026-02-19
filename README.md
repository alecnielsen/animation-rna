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

# Full ribosome render (slow, only after style is locked)
python3.11 render.py
```

Output goes to `renders/`.

## Visual style spec

See [STYLE.md](STYLE.md) for the full visual design specification.

## Animation plan

See [PLAN.md](PLAN.md) for the full animation plan and technical approach.
