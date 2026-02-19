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

## PDB structure

Using [6Y0G](https://www.rcsb.org/structure/6Y0G) — human 80S ribosome in the classical PRE state (3.2 A cryo-EM). Contains:

| Chain(s) | Component |
|----------|-----------|
| `S*` (34 chains) | 40S small subunit (18S rRNA + proteins) |
| `L*` (45 chains) | 60S large subunit (28S/5.8S/5S rRNA + proteins) |
| `A4` | mRNA |
| `B4` | P-site tRNA |
| `D4` | A-site tRNA |
| `C4` | Nascent peptide |

## Rendering

Scripts run headlessly via the `bpy` Python module — no Blender GUI needed.

```bash
source mn_env/bin/activate

# Test render (single frame)
python3.11 test_render.py

# Full animation (TBD)
python3.11 render.py
```

Output goes to `renders/`.

## Animation plan

Rigid-body choreography using components from 6Y0G. One elongation cycle, looped (~10-15s):

1. Ribosome with mRNA threaded, P-site tRNA holding peptide
2. Aminoacyl-tRNA enters A-site
3. Peptide transfers to A-site tRNA
4. Translocation: tRNAs shift A→P→E, mRNA advances one codon
5. Polypeptide chain grows by one residue
6. Loop

Target style: cartoon ribbons with semi-transparent surface overlay, colored by component.
