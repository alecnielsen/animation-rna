"""Render a single high-quality frame of the ribosome translation complex.

Loads all molecules at their crystallographic/built coordinates and renders
a single static frame. No animation logic, no jitter, no PCA, no progressive
reveal.

Components:
  1. Ribosome (40S + 60S): translucent surface from 6Y0G
  2. Extended mRNA: blue surface from extended_mrna.pdb
  3. P-site tRNA: orange surface, chain B4 from 6Y0G
  4. A-site tRNA: orange surface, chain D4 from 6Y0G (not B4!)
  5. Polypeptide: magenta surface from tunnel_polypeptide.pdb

Output: renders/single_frame.png

Run with:
  python3.11 render_single_frame.py          # 1920x1080, 128 samples
  python3.11 render_single_frame.py --debug   # 960x540, 32 samples
"""

import molecularnodes as mn
import bpy
import numpy as np
import os
import sys
import math

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = "--debug" in sys.argv

if DEBUG:
    RES = (960, 540)
    SAMPLES = 32
else:
    RES = (1920, 1080)
    SAMPLES = 128

OUTPUT_DIR = "renders"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "single_frame.png")

# Chain definitions
CHAINS_40S = [
    "S2", "SA", "SB", "SC", "SD", "SE", "SF", "SG", "SH", "SI", "SJ", "SK",
    "SL", "SM", "SN", "SO", "SP", "SQ", "SR", "SS", "ST", "SU", "SV", "SW",
    "SX", "SY", "SZ", "Sa", "Sb", "Sc", "Sd", "Se", "Sf", "Sg",
]
CHAINS_60S = [
    "L5", "L7", "L8", "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LI",
    "LJ", "LL", "LM", "LN", "LO", "LP", "LQ", "LR", "LS", "LT", "LU", "LV",
    "LW", "LX", "LY", "LZ", "La", "Lb", "Lc", "Ld", "Le", "Lg", "Lh",
    "Li", "Lj", "Lk", "Ll", "Lm", "Ln", "Lo", "Lp", "Lr",
]
RIBOSOME_CHAINS = CHAINS_40S + CHAINS_60S


# ---------------------------------------------------------------------------
# Material helpers (reused from animate.py)
# ---------------------------------------------------------------------------
def make_solid_material(color):
    mat = bpy.data.materials.new(name="solid")
    n = mat.node_tree.nodes
    l = mat.node_tree.links
    n.clear()
    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    bsdf.inputs["Emission Color"].default_value = (*color, 1.0)
    bsdf.inputs["Emission Strength"].default_value = 0.8
    out = n.new("ShaderNodeOutputMaterial")
    l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_translucent_surface_material():
    """Translucent material for the ribosome using Principled BSDF Alpha."""
    mat = bpy.data.materials.new(name="translucent_surface")
    mat.blend_method = 'HASHED'
    n = mat.node_tree.nodes
    l = mat.node_tree.links
    n.clear()

    bsdf = n.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.45, 0.55, 0.75, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Alpha"].default_value = 0.06

    out = n.new("ShaderNodeOutputMaterial")
    l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def set_bg(scene, color, strength):
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*color, 1.0)
        bg.inputs["Strength"].default_value = strength


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mn.register()

    mode = "DEBUG" if DEBUG else "FULL"
    print(f"=== Rendering single frame ({mode}: {RES[0]}x{RES[1]}, "
          f"{SAMPLES} samples) ===")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    canvas = mn.Canvas(mn.scene.Cycles(samples=SAMPLES), resolution=RES)
    scene = bpy.context.scene
    scene.cycles.device = 'CPU'  # avoid GPU hang on macOS
    scene.render.film_transparent = False
    set_bg(scene, (0.04, 0.04, 0.06), 0.5)
    scene.cycles.max_bounces = 12
    scene.cycles.transparent_max_bounces = 64

    # --- Load molecules ---
    print("  Loading molecules...")

    # 1. Ribosome surface (40S + 60S) -- translucent
    mol_surface = mn.Molecule.fetch("6Y0G")
    mol_surface.add_style(
        style=mn.StyleSurface(),
        selection=mol_surface.select.chain_id(RIBOSOME_CHAINS),
        material=make_translucent_surface_material(),
        name="surface",
    )

    # 2. mRNA (extended, from preprocessed PDB)
    mol_mrna = mn.Molecule.load("extended_mrna.pdb")
    mol_mrna.add_style(
        style=mn.StyleSurface(),
        material=make_solid_material((0.1, 0.35, 0.95)),
        name="mRNA",
    )

    # 3. P-site tRNA (chain B4)
    mol_trna_p = mn.Molecule.fetch("6Y0G")
    mol_trna_p.add_style(
        style=mn.StyleSurface(),
        selection=mol_trna_p.select.chain_id(["B4"]),
        material=make_solid_material((0.95, 0.5, 0.1)),
        name="tRNA_P",
    )

    # 4. A-site tRNA (chain D4 -- NOT B4!)
    mol_trna_a = mn.Molecule.fetch("6Y0G")
    mol_trna_a.add_style(
        style=mn.StyleSurface(),
        selection=mol_trna_a.select.chain_id(["D4"]),
        material=make_solid_material((0.95, 0.5, 0.1)),
        name="tRNA_A",
    )

    # 5. Polypeptide (tunnel-threaded)
    peptide_pdb = "tunnel_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        print(f"  WARNING: {peptide_pdb} not found, falling back to extended_polypeptide.pdb")
        peptide_pdb = "extended_polypeptide.pdb"
    mol_peptide = mn.Molecule.load(peptide_pdb)
    mol_peptide.add_style(
        style=mn.StyleSurface(),
        material=make_solid_material((0.8, 0.15, 0.6)),
        name="polypeptide",
    )

    # --- Find Blender objects ---
    objs_6y0g = sorted(
        [o for o in bpy.data.objects if "6Y0G" in o.name and o.type == "MESH"],
        key=lambda o: o.name,
    )
    mrna_search = "extended_mrna"
    pep_search = os.path.splitext(os.path.basename(peptide_pdb))[0]
    objs_mrna = [o for o in bpy.data.objects if mrna_search in o.name and o.type == "MESH"]
    objs_pep = [o for o in bpy.data.objects if pep_search in o.name and o.type == "MESH"]

    print(f"  Found {len(objs_6y0g)} 6Y0G objects: {[o.name for o in objs_6y0g]}")
    print(f"  Found {len(objs_mrna)} mRNA objects: {[o.name for o in objs_mrna]}")
    print(f"  Found {len(objs_pep)} polypeptide objects: {[o.name for o in objs_pep]}")

    if len(objs_6y0g) < 3 or len(objs_mrna) < 1 or len(objs_pep) < 1:
        print(f"  ERROR: Expected 3 6Y0G + 1 mRNA + 1 polypeptide objects")
        return

    obj_surface = objs_6y0g[0]
    obj_trna_p = objs_6y0g[1]
    obj_trna_a = objs_6y0g[2]
    obj_mrna = objs_mrna[0]
    obj_peptide = objs_pep[0]

    # Apply z-rotation to all primary objects (matches animate.py orientation)
    for o in [obj_surface, obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]:
        o.rotation_euler.z = math.pi / 2

    # --- Camera setup ---
    canvas.frame_object(mol_surface)
    print(f"  Camera: loc={tuple(scene.camera.location)}, lens={scene.camera.data.lens}")

    bpy.context.view_layer.update()

    # --- Render ---
    print(f"  Rendering to {OUTPUT_FILE}...")
    canvas.snapshot(OUTPUT_FILE)
    print(f"  Saved: {OUTPUT_FILE}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
