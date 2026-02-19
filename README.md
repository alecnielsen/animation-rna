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

## PDB structures

| PDB ID | Description | Use |
|--------|-------------|-----|
| [4V6F](https://www.rcsb.org/structure/4V6F) | 70S ribosome with mRNA + A/P/E-site tRNAs (3.1 A) | Primary elongation complex |
| [4V7B](https://www.rcsb.org/structure/4V7B) | tRNAs in transit during EF-G translocation | Translocation intermediate |
| [6Y0G](https://www.rcsb.org/structure/6Y0G) | Human 80S ribosome, classical PRE state (3.2 A) | Eukaryotic alternative |

## Rendering

Scripts run headlessly via the `bpy` Python module -- no Blender GUI needed.

```bash
source mn_env/bin/activate
python3.11 render.py
```

Output goes to `renders/`.
