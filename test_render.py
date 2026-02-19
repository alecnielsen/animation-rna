"""
Test script: Load human 80S ribosome (6Y0G) and render with per-component styling.
- Ribosome: near-transparent surface with outline
- mRNA, tRNAs, polypeptide: solid ball-and-stick

Run with: python3.11 test_render.py
"""

import molecularnodes as mn
import bpy
import os

os.makedirs("renders", exist_ok=True)
mn.register()

# --- Chain ID definitions ---

CHAINS_40S = [
    "S2", "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK",
    "SL", "SM", "SN", "SO", "SP", "SQ", "SR", "SS", "ST", "SU", "SV", "SW",
    "SX", "SY", "SZ", "Sa", "Sb", "Sc", "Sd", "Se", "Sf", "Sg",
]

CHAINS_60S = [
    "L5", "L7", "L8", "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LI",
    "LJ", "LL", "LM", "LN", "LO", "LP", "LQ", "LR", "LS", "LT", "LU", "LV",
    "LW", "LX", "LY", "LZ", "La", "Lb", "Lc", "Ld", "Le", "Lf", "Lg", "Lh",
    "Li", "Lj", "Lk", "Ll", "Lm", "Ln", "Lo", "Lp", "Lr",
]

CHAINS_RIBOSOME = CHAINS_40S + CHAINS_60S

CHAIN_MRNA = "A4"
CHAIN_TRNA_P = "B4"
CHAIN_TRNA_A = "D4"
CHAIN_PEPTIDE = "C4"
CHAINS_PAYLOAD = [CHAIN_MRNA, CHAIN_TRNA_P, CHAIN_TRNA_A, CHAIN_PEPTIDE]

# --- Setup ---

print("Setting up canvas...")
canvas = mn.Canvas(
    mn.scene.Cycles(samples=64),
    resolution=(1920, 1080),
)

print("Fetching 6Y0G (human 80S ribosome)...")
mol = mn.Molecule.fetch("6Y0G")

# Debug: print actual chain IDs in the structure
unique_chains = set(mol.array.chain_id) if hasattr(mol.array, 'chain_id') else set()
print(f"Chain IDs in structure ({len(unique_chains)}): {sorted(unique_chains)}")

# --- Style: Ribosome — transparent surface with outline ---

print("Applying ribosome surface style (transparent)...")
mol.add_style(
    style=mn.StyleSurface(),
    selection=mol.select.chain_id(CHAINS_RIBOSOME),
    material=mn.material.TransparentOutline(),
    name="ribosome_surface",
)

# --- Style: mRNA — ball and stick, blue ---

print("Applying mRNA style...")
mol.add_style(
    style=mn.StyleBallAndStick(),
    selection=mol.select.chain_id([CHAIN_MRNA]),
    material=mn.material.Default(),
    name="mrna",
)

# --- Style: tRNAs — ball and stick ---

print("Applying tRNA styles...")
mol.add_style(
    style=mn.StyleBallAndStick(),
    selection=mol.select.chain_id([CHAIN_TRNA_P, CHAIN_TRNA_A]),
    material=mn.material.Default(),
    name="trnas",
)

# --- Style: Nascent peptide — ball and stick ---

print("Applying peptide style...")
mol.add_style(
    style=mn.StyleBallAndStick(),
    selection=mol.select.chain_id([CHAIN_PEPTIDE]),
    material=mn.material.Default(),
    name="peptide",
)

# --- Render ---

print("Framing and rendering...")
canvas.frame_object(mol)
canvas.snapshot("renders/test_styled.png")

print("Done! Check renders/test_styled.png")
