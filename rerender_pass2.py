"""Re-render only pass 2 (surface) with transparent background for alpha mask."""
import molecularnodes as mn
import bpy
import math

mn.register()

RES = (1920, 1080)

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


def make_surface_material():
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


canvas2 = mn.Canvas(mn.scene.Cycles(samples=16), resolution=RES)
scene2 = bpy.context.scene
set_bg(scene2, (0.02, 0.02, 0.03), 0.3)
scene2.render.film_transparent = True

mol2 = mn.Molecule.fetch("6Y0G")
mol2.add_style(
    style=mn.StyleSurface(),
    selection=mol2.select.chain_id(RIBOSOME_CHAINS),
    material=make_surface_material(),
    name="surface",
)

for o in bpy.data.objects:
    if "6Y0G" in o.name and o.type == "MESH":
        o.rotation_euler.z = math.pi / 2

canvas2.frame_object(mol2)

# Save camera for pass 1 reuse
cam = scene2.camera
print(f"Camera: loc={tuple(cam.location)}")

canvas2.snapshot("renders/pass2_ribosome_surface.png")
print("Saved pass 2 with transparent background")
