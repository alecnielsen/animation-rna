"""
Style development: Two-pass compositing.
Frame camera on the SURFACE (larger object) so outline is centered.

Run with: python3.11 test_render.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from PIL import Image, ImageFilter
import os

os.makedirs("renders", exist_ok=True)
mn.register()

RES = (1920, 1080)
OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 3


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
# PASS 2 FIRST — Surface mask (for camera framing)
# ==========================================
print("=== Pass 2: Surface mask (render first for framing) ===")
canvas2 = mn.Canvas(mn.scene.Cycles(samples=16), resolution=RES)
scene2 = bpy.context.scene
set_bg(scene2, (0.02, 0.02, 0.03), 0.3)

mol2 = mn.Molecule.fetch("2PTC")
mol2.add_style(
    style=mn.StyleSurface(),
    selection=mol2.select.chain_id(["E"]),
    material=make_surface_material(),
    name="surface",
)

# Rotate the molecule 90 degrees around Z axis BEFORE framing
import math
for o in bpy.data.objects:
    if "2PTC" in o.name and o.type == "MESH":
        o.rotation_euler.z = math.pi / 2

# Frame on the surface — this is the larger object
canvas2.frame_object(mol2)

# Capture camera
cam = scene2.camera
cam_loc = tuple(cam.location)
cam_rot = tuple(cam.rotation_euler)
cam_lens = cam.data.lens
cam_clip = (cam.data.clip_start, cam.data.clip_end)
print(f"  Camera: loc={cam_loc}")

canvas2.snapshot("renders/pass2_surface.png")
print("  Saved")
canvas2.clear()

# ==========================================
# PASS 1 — Atoms with same camera
# ==========================================
print("=== Pass 1: Atoms ===")
canvas1 = mn.Canvas(mn.scene.Cycles(samples=48), resolution=RES)
scene1 = bpy.context.scene
set_bg(scene1, (0.04, 0.04, 0.06), 0.5)
scene1.cycles.max_bounces = 12

mol1 = mn.Molecule.fetch("2PTC")
mol1.add_style(
    style=mn.StyleCartoon(),
    selection=mol1.select.chain_id(["I"]),
    material=make_solid_material((0.1, 0.35, 0.95)),
    name="atoms",
)

# Apply same rotation to pass 1 molecule
for o in bpy.data.objects:
    if "2PTC" in o.name and o.type == "MESH":
        o.rotation_euler.z = math.pi / 2

# Apply same camera
cam1 = scene1.camera
cam1.location = cam_loc
cam1.rotation_euler = cam_rot
cam1.data.lens = cam_lens
cam1.data.clip_start, cam1.data.clip_end = cam_clip

canvas1.snapshot("renders/pass1_atoms.png")
print("  Saved")

# ==========================================
# COMPOSITE
# ==========================================
print("=== Compositing ===")

SURFACE_OPACITY = 0.20  # 20% translucent overlay

atoms = Image.open("renders/pass1_atoms.png").convert("RGBA")
surface = Image.open("renders/pass2_surface.png").convert("RGBA")
surface_gray = Image.open("renders/pass2_surface.png").convert("L")

# --- Layer 1: Translucent surface overlay ---
surface_np = np.array(surface).astype(np.float32)
surface_np[:, :, 3] = SURFACE_OPACITY * 255
translucent = Image.fromarray(surface_np.astype(np.uint8), "RGBA")
result = Image.alpha_composite(atoms, translucent)

# --- Layer 2: Edge outline (thin, crisp, with interior detail) ---
edges = surface_gray.filter(ImageFilter.FIND_EDGES)
edges_np = np.array(edges)
edges_binary = (edges_np > 15).astype(np.uint8) * 255

edges_img = Image.fromarray(edges_binary)
for _ in range(OUTLINE_THICKNESS // 2):
    edges_img = edges_img.filter(ImageFilter.MaxFilter(3))
edges_np = np.array(edges_img)

overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
mask = edges_np > 100
overlay[mask, 0] = OUTLINE_COLOR[0]
overlay[mask, 1] = OUTLINE_COLOR[1]
overlay[mask, 2] = OUTLINE_COLOR[2]
overlay[mask, 3] = 255

result = Image.alpha_composite(result, Image.fromarray(overlay, "RGBA"))
result.save("renders/test_style.png")

print("Done! Check renders/test_style.png")
