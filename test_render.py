"""
Test script: Load human 80S ribosome (6Y0G) and render a single still frame.
Run with: python3.11 test_render.py
"""

import molecularnodes as mn
import bpy
import os

os.makedirs("renders", exist_ok=True)

# Register MN with the Blender scene
mn.register()

# Set up render canvas first (it creates a clean scene)
print("Setting up render canvas...")
canvas = mn.Canvas(
    mn.scene.Cycles(samples=64),
    resolution=(1920, 1080),
)

print("Fetching 6Y0G (human 80S ribosome)... this may take a few minutes.")
mol = mn.Molecule.fetch("6Y0G")

print("Applying cartoon style...")
mol.add_style(
    mn.StyleCartoon(),
    material=mn.material.AmbientOcclusion(),
)

print("Framing and rendering...")
canvas.frame_object(mol)
canvas.snapshot("renders/test_frame.png")

print("Done! Check renders/test_frame.png")
