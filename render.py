"""
Full ribosome render: 6Y0G human 80S ribosome with two-pass compositing.

Pass 1: Cartoon internal components (mRNA, tRNAs, polypeptide)
Pass 2: Surface mesh of ribosome (40S + 60S subunits)
Composite: Translucent surface overlay + edge outline

Run with: python3.11 render.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from PIL import Image, ImageFilter
import os
import math

os.makedirs("renders", exist_ok=True)
mn.register()

RES = (1920, 1080)
OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 3
SURFACE_OPACITY = 0.20

# 6Y0G chain IDs
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
RIBOSOME_CHAINS = CHAINS_40S + CHAINS_60S
INTERNAL_CHAINS = ["A4", "B4", "C4", "D4"]


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


def make_surface_material():
    """Flat diffuse surface — shading variations create interior detail in outline."""
    mat = bpy.data.materials.new(name="surface_flat")
    n = mat.node_tree.nodes
    l = mat.node_tree.links
    n.clear()
    bsdf = n.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (0.45, 0.55, 0.75, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    out = n.new("ShaderNodeOutputMaterial")
    l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def set_bg(scene, color, strength):
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*color, 1.0)
        bg.inputs["Strength"].default_value = strength


# ==========================================
# PASS 2 FIRST — Surface (for camera framing)
# ==========================================
print("=== Pass 2: Ribosome surface (render first for framing) ===")
canvas2 = mn.Canvas(mn.scene.Cycles(samples=16), resolution=RES)
scene2 = bpy.context.scene
set_bg(scene2, (0.02, 0.02, 0.03), 0.3)
# Transparent background so alpha channel = object mask (for silhouette detection)
scene2.render.film_transparent = True

mol2 = mn.Molecule.fetch("6Y0G")
mol2.add_style(
    style=mn.StyleSurface(),
    selection=mol2.select.chain_id(RIBOSOME_CHAINS),
    material=make_surface_material(),
    name="surface",
)

# Rotate for a good viewing angle
for o in bpy.data.objects:
    if "6Y0G" in o.name and o.type == "MESH":
        o.rotation_euler.z = math.pi / 2

# Frame on the ribosome surface
canvas2.frame_object(mol2)

# Capture camera transform
cam = scene2.camera
cam_loc = tuple(cam.location)
cam_rot = tuple(cam.rotation_euler)
cam_lens = cam.data.lens
cam_clip = (cam.data.clip_start, cam.data.clip_end)
print(f"  Camera: loc={cam_loc}")

canvas2.snapshot("renders/pass2_ribosome_surface.png")
print("  Saved pass 2")
canvas2.clear()

# ==========================================
# PASS 1 — Internal components with same camera
# ==========================================
print("=== Pass 1: Internal components (mRNA, tRNAs, polypeptide) ===")
canvas1 = mn.Canvas(mn.scene.Cycles(samples=48), resolution=RES)
scene1 = bpy.context.scene
set_bg(scene1, (0.04, 0.04, 0.06), 0.5)
scene1.cycles.max_bounces = 12

mol1 = mn.Molecule.fetch("6Y0G")

# mRNA — vibrant blue cartoon
mol1.add_style(
    style=mn.StyleCartoon(),
    selection=mol1.select.chain_id(["A4"]),
    material=make_solid_material((0.1, 0.35, 0.95)),
    name="mRNA",
)

# tRNAs — vibrant orange cartoon
mol1.add_style(
    style=mn.StyleCartoon(),
    selection=mol1.select.chain_id(["B4", "D4"]),
    material=make_solid_material((0.95, 0.5, 0.1)),
    name="tRNAs",
)

# Nascent polypeptide — magenta cartoon
mol1.add_style(
    style=mn.StyleCartoon(),
    selection=mol1.select.chain_id(["C4"]),
    material=make_solid_material((0.8, 0.15, 0.6)),
    name="polypeptide",
)

# Apply same rotation
for o in bpy.data.objects:
    if "6Y0G" in o.name and o.type == "MESH":
        o.rotation_euler.z = math.pi / 2

# Apply same camera
cam1 = scene1.camera
cam1.location = cam_loc
cam1.rotation_euler = cam_rot
cam1.data.lens = cam_lens
cam1.data.clip_start, cam1.data.clip_end = cam_clip

canvas1.snapshot("renders/pass1_internal.png")
print("  Saved pass 1")

# ==========================================
# COMPOSITE
# ==========================================
print("=== Compositing ===")

atoms = Image.open("renders/pass1_internal.png").convert("RGBA")
surface = Image.open("renders/pass2_ribosome_surface.png").convert("RGBA")
surface_gray = Image.open("renders/pass2_ribosome_surface.png").convert("L")

# Layer 1: Translucent surface overlay
surface_np = np.array(surface).astype(np.float32)
surface_np[:, :, 3] = SURFACE_OPACITY * 255
translucent = Image.fromarray(surface_np.astype(np.uint8), "RGBA")
result = Image.alpha_composite(atoms, translucent)

# Layer 2: Outer silhouette from alpha channel (transparent bg render)
alpha = np.array(surface)[:, :, 3]
alpha_mask = (alpha > 10).astype(np.uint8) * 255
mask_img = Image.fromarray(alpha_mask)
mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
mask_img = Image.fromarray((np.array(mask_img) > 128).astype(np.uint8) * 255)
silhouette = mask_img.filter(ImageFilter.FIND_EDGES)
sil_np = (np.array(silhouette) > 30).astype(np.uint8) * 255
sil_img = Image.fromarray(sil_np)
for _ in range(OUTLINE_THICKNESS // 2):
    sil_img = sil_img.filter(ImageFilter.MaxFilter(3))

edges_np = np.array(sil_img)
overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
mask = edges_np > 100
overlay[mask, 0] = OUTLINE_COLOR[0]
overlay[mask, 1] = OUTLINE_COLOR[1]
overlay[mask, 2] = OUTLINE_COLOR[2]
overlay[mask, 3] = 255

result = Image.alpha_composite(result, Image.fromarray(overlay, "RGBA"))
result.save("renders/ribosome_style.png")

print("Done! Check renders/ribosome_style.png")
