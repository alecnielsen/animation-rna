"""Render a single high-quality frame of the ribosome translation complex.

Loads all molecules at their crystallographic/built coordinates and renders
a single static frame. No animation logic, no jitter, no PCA, no progressive
reveal.

Applies mRNA bend (quadratic droop outside ribosome channel, copied from
animate.py) for organic curvature.

Camera angle
------------
Orthographic, manually found in Blender and exported via View > Align Active
Camera to View. The rotation is euler XYZ (2.2480, 0.0, 0.0489) rad — this
gives a slightly-below-horizontal view looking up at the ribosome, rotated
~2.8° around Z. The orbit center (camera target) is (-2.66, 1.71, 1.72) BU.

Key property: the nascent polypeptide exits the ribosome tunnel *parallel*
to the viewing plane (left-right in the frame, not into/out of screen).

To find a new angle:
  1. python3.11 render_single_frame.py --save-blend
  2. Open scene.blend in Blender, orbit to desired angle
  3. View > Align Active Camera to View, then Cmd+S
  4. Read camera params:
       python3.11 -c "import bpy; bpy.ops.wm.open_mainfile(filepath='scene.blend'); \\
         c=bpy.context.scene.camera; print(f'rot={tuple(c.rotation_euler)}'); \\
         r=[a.spaces[0].region_3d for a in bpy.context.screen.areas if a.type=='VIEW_3D'][0]; \\
         print(f'target={tuple(r.view_location)}')"
  5. Update cam_rot and target in the Camera setup section below.

Components:
  1. Ribosome (40S + 60S): cartoon outline (2-pass composite) from 6Y0G
  2. Extended mRNA: blue cartoon from extended_mrna.pdb (with droop bend)
  3. P-site tRNA: orange ribbon, chain B4 from 6Y0G
  4. A-site tRNA: orange ribbon, chain D4 from 6Y0G (not B4!)
  5. Polypeptide: magenta spheres from tunnel_polypeptide.pdb

Output: renders/single_frame.png (or scene.blend with --save-blend)

Run with:
  python3.11 render_single_frame.py               # 1920x1080, 128 samples, cartoon style
  python3.11 render_single_frame.py --debug        # 960x540, 32 samples
  python3.11 render_single_frame.py --gpu          # CUDA GPU rendering
  python3.11 render_single_frame.py --save-blend   # Build scene, save as scene.blend (no render)
  python3.11 render_single_frame.py --style=surface # Production quality (slow, ~15 min setup)
  python3.11 render_single_frame.py --style=cartoon # Fast preview (default, ~10s setup)
"""

import molecularnodes as mn
import bpy
import numpy as np
import os
import sys
import math
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = "--debug" in sys.argv
GPU = "--gpu" in sys.argv
SAVE_BLEND = "--save-blend" in sys.argv

# Molecule style: cartoon (fast, ~1s) or surface (slow, ~15min but production quality)
MOL_STYLE = "cartoon"
for arg in sys.argv:
    if arg.startswith("--style="):
        MOL_STYLE = arg.split("=", 1)[1]

# Ribosome style: hashed (default), volume, sss (only used with --style=surface)
RIBOSOME_STYLE = "hashed"
for arg in sys.argv:
    if arg.startswith("--ribosome-style="):
        RIBOSOME_STYLE = arg.split("=", 1)[1]

if DEBUG:
    RES = (960, 540)
    SAMPLES = 32
else:
    RES = (1920, 1080)
    SAMPLES = 128

OUTPUT_DIR = "renders"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "single_frame.png")
BLEND_FILE = "scene.blend"

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
# mRNA bend constants + function (from animate.py)
# ---------------------------------------------------------------------------
MRNA_CHANNEL_HALF_LEN = 4.0   # BU — straight zone around mRNA centroid
MRNA_BEND_STRENGTH = 0.015    # BU per BU² beyond channel (quadratic droop)


def get_mesh_res_ids(obj):
    """Read per-vertex res_id from MN mesh attributes."""
    mesh = obj.data
    n = len(mesh.vertices)
    for attr_name in ['res_id', 'residue_id']:
        if attr_name in mesh.attributes:
            res_ids = np.zeros(n, dtype=np.int32)
            mesh.attributes[attr_name].data.foreach_get('value', res_ids)
            return res_ids
    return None


def apply_mrna_bend(positions, res_ids):
    """Apply a gentle quadratic droop to mRNA vertices outside the ribosome channel.

    Vertices within +/-MRNA_CHANNEL_HALF_LEN of the mRNA centroid along the
    principal axis stay straight. Beyond that, they droop quadratically
    perpendicular to the axis, giving the mRNA an organic curved look
    instead of a rigid rod.

    The principal axis is computed via SVD on the actual vertex positions
    (local/object space), avoiding coordinate-space mismatches with any
    world-space constants.

    Modifies positions in-place and returns them.
    """
    centroid = positions.mean(axis=0)
    relative = positions - centroid

    # Compute principal axis from vertex positions via SVD (local space)
    _, _, vt = np.linalg.svd(relative, full_matrices=False)
    local_axis = vt[0]  # first principal component = mRNA long axis

    # Project onto mRNA axis
    proj = relative @ local_axis  # scalar projection per vertex

    # Droop direction: perpendicular to local_axis in the XY plane
    z_up = np.array([0.0, 0.0, 1.0])
    droop_dir = np.cross(local_axis, z_up)
    droop_norm = np.linalg.norm(droop_dir)
    if droop_norm < 1e-6:
        droop_dir = np.cross(local_axis, np.array([1.0, 0.0, 0.0]))
        droop_norm = np.linalg.norm(droop_dir)
    droop_dir = droop_dir / droop_norm

    # Also add a Z component for gravity-like sag
    droop_dir = 0.7 * droop_dir + 0.3 * np.array([0.0, 0.0, -1.0])
    droop_dir = droop_dir / np.linalg.norm(droop_dir)

    # Apply quadratic droop beyond the straight zone
    for i in range(len(positions)):
        d = abs(proj[i]) - MRNA_CHANNEL_HALF_LEN
        if d > 0:
            droop = MRNA_BEND_STRENGTH * d * d
            positions[i] += droop * droop_dir

    return positions


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
    bsdf.inputs["Emission Strength"].default_value = 1.2
    out = n.new("ShaderNodeOutputMaterial")
    l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_translucent_surface_material(style=None):
    """Translucent material for the ribosome.

    Styles:
      hashed  — Principled BSDF Alpha=0.12 + HASHED blend (denoiser smooths noise)
      volume  — Volume Absorption shader (physically-based, thin=transparent)
      sss     — Principled BSDF with Subsurface Scattering (waxy translucent)
    """
    if style is None:
        style = RIBOSOME_STYLE

    mat = bpy.data.materials.new(name="translucent_surface")
    n = mat.node_tree.nodes
    l = mat.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")

    if style == "volume":
        # Volume Absorption: physically-based translucency where thin parts
        # are more see-through. No surface shader — pure volume.
        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.55, 0.65, 0.85, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Alpha"].default_value = 0.3
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        mat.blend_method = 'HASHED'

        absorb = n.new("ShaderNodeVolumeAbsorption")
        absorb.inputs["Color"].default_value = (0.7, 0.8, 0.95, 1.0)
        absorb.inputs["Density"].default_value = 0.15
        l.new(absorb.outputs["Volume"], out.inputs["Volume"])

    elif style == "sss":
        # Subsurface Scattering: smooth, waxy translucent look.
        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.45, 0.55, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Subsurface Weight"].default_value = 0.8
        bsdf.inputs["Subsurface Radius"].default_value = (0.5, 0.5, 0.5)
        bsdf.inputs["Subsurface Scale"].default_value = 0.5
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    else:  # "hashed" (default)
        # Higher alpha (0.12) with denoiser — smoother than 0.06
        mat.blend_method = 'HASHED'
        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.45, 0.55, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Alpha"].default_value = 0.12
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    return mat


def set_bg(scene, color, strength):
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (*color, 1.0)
        bg.inputs["Strength"].default_value = strength


def _write_backbone(in_pdb, out_pdb, mol_type="rna"):
    """Write a backbone-only PDB for cleaner visualization.

    RNA backbone: P, O5', C5', C4', C3', O3'
    Protein backbone: N, CA, C, O
    """
    from biotite.structure.io.pdb import PDBFile as BiotitePDB
    from biotite.structure import AtomArrayStack

    pdb = BiotitePDB.read(in_pdb)
    arr = pdb.get_structure(model=1)
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    if mol_type == "rna":
        bb_names = {"P", "O5'", "C5'", "C4'", "C3'", "O3'"}
    else:
        bb_names = {"N", "CA", "C", "O"}

    bb = arr[np.isin(arr.atom_name, list(bb_names))]
    out = BiotitePDB()
    out.set_structure(bb)
    out.write(out_pdb)
    print(f"  Backbone: {in_pdb} ({len(arr)} atoms) -> {out_pdb} ({len(bb)} atoms)")


def _extract_trna_pdbs():
    """Extract tRNA chains from 6Y0G as separate PDB files if not already cached.

    This avoids loading the full 210K-atom 6Y0G structure multiple times
    just to select ~2K-atom tRNA chains. Files are written once and reused.

    6Y0G uses 2-character chain IDs (B4, D4) which PDB format can't handle,
    so we remap to single-character IDs (A) before writing.
    """
    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile as BiotitePDB
    import biotite.structure.io.pdbx as pdbx
    from pathlib import Path

    chains_to_extract = {"B4": "trna_b4.pdb", "D4": "trna_d4.pdb"}

    # Skip if all files already exist
    if all(os.path.exists(f) for f in chains_to_extract.values()):
        print("  tRNA PDBs already cached")
        return

    # Load from MN's cache or fetch
    cache_dir = Path.home() / "MolecularNodesCache"
    bcif_path = cache_dir / "6Y0G.bcif"
    if not bcif_path.exists():
        print("  Fetching 6Y0G for tRNA extraction...")
        import biotite.database.rcsb as rcsb_db
        rcsb_db.fetch("6Y0G", "bcif", target_path=str(cache_dir))

    print("  Extracting tRNA chains from cached 6Y0G...")
    cif = pdbx.BinaryCIFFile.read(str(bcif_path))
    arr = pdbx.get_structure(cif, model=1)
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    for chain_id, filename in chains_to_extract.items():
        chain_arr = arr[arr.chain_id == chain_id].copy()
        # Remap 2-char chain ID to "A" for PDB format compatibility
        chain_arr.chain_id[:] = "A"
        pdb_file = BiotitePDB()
        pdb_file.set_structure(chain_arr)
        pdb_file.write(filename)
        print(f"    {filename}: {len(chain_arr)} atoms (chain {chain_id} -> A)")


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

    if GPU:
        # Enable CUDA GPU rendering
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        cycles_prefs.get_devices()
        for device in cycles_prefs.devices:
            device.use = True
        scene.cycles.device = 'GPU'
        print(f"  GPU rendering enabled (CUDA)")
    else:
        scene.cycles.device = 'CPU'  # avoid GPU hang on macOS
    scene.render.film_transparent = False
    set_bg(scene, (0.04, 0.04, 0.06), 0.5)
    scene.cycles.max_bounces = 12
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.use_denoising = True  # OpenImageDenoise smooths hashed noise

    # --- Load molecules ---
    print("  Loading molecules...")

    # Extract tRNA chains as separate PDB files (avoids processing all 210K
    # atoms of 6Y0G three times — tRNAs are only ~2K atoms each)
    _extract_trna_pdbs()

    # Style helpers
    def ribo_style():
        if MOL_STYLE == "surface":
            return mn.StyleSurface()
        return "cartoon"  # ribosome always cartoon (fast for 210K atoms)

    def detail_style():
        if MOL_STYLE == "surface":
            return mn.StyleSurface()
        return "ribbon"  # mRNA/tRNA/peptide: ribbon style

    # Flat opaque material for ribosome silhouette pass (rendered with
    # film_transparent=True, then PIL edge-detects the alpha channel)
    def make_ribo_material():
        if MOL_STYLE == "surface":
            return make_translucent_surface_material()
        mat = bpy.data.materials.new(name="ribo_flat")
        n = mat.node_tree.nodes
        l = mat.node_tree.links
        n.clear()
        bsdf = n.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs["Color"].default_value = (0.45, 0.55, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 1.0
        out = n.new("ShaderNodeOutputMaterial")
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
        return mat

    # 1. Ribosome (40S + 60S) — cartoon with transparency
    mol_surface = mn.Molecule.fetch("6Y0G")
    mol_surface.add_style(
        style=ribo_style(),
        selection=mol_surface.select.chain_id(RIBOSOME_CHAINS),
        material=make_ribo_material(),
        name="surface",
    )

    # 2. mRNA (extended, from preprocessed PDB) — backbone cartoon (gaps = breaks)
    _write_backbone("extended_mrna.pdb", "extended_mrna_bb.pdb", mol_type="rna")
    mol_mrna = mn.Molecule.load("extended_mrna_bb.pdb")
    mol_mrna.add_style(
        style="cartoon" if MOL_STYLE != "surface" else mn.StyleSurface(),
        material=make_solid_material((0.05, 0.25, 0.95)),
        name="mRNA",
    )

    # 3. P-site tRNA (chain B4 — extracted PDB, ~2K atoms)
    mol_trna_p = mn.Molecule.load("trna_b4.pdb")
    mol_trna_p.add_style(
        style=detail_style(),
        material=make_solid_material((0.95, 0.4, 0.0)),
        name="tRNA_P",
    )

    # 4. A-site tRNA (chain D4 — extracted PDB, ~2K atoms)
    mol_trna_a = mn.Molecule.load("trna_d4.pdb")
    mol_trna_a.add_style(
        style=detail_style(),
        material=make_solid_material((0.95, 0.4, 0.0)),
        name="tRNA_A",
    )

    # 5. Polypeptide (tunnel-threaded, repeating domains) — spheres
    peptide_pdb = "repeating_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        peptide_pdb = "tunnel_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        print(f"  WARNING: no polypeptide PDB found, falling back to extended_polypeptide.pdb")
        peptide_pdb = "extended_polypeptide.pdb"
    mol_peptide = mn.Molecule.load(peptide_pdb)
    mol_peptide.add_style(
        style="spheres" if MOL_STYLE != "surface" else mn.StyleSurface(),
        material=make_solid_material((0.85, 0.05, 0.55)),
        name="polypeptide",
    )

    # --- Find Blender objects ---
    def find_mesh(name_substr):
        return [o for o in bpy.data.objects if name_substr in o.name and o.type == "MESH"]

    objs_surface = find_mesh("6Y0G")
    objs_mrna = find_mesh("extended_mrna")
    objs_trna_p = find_mesh("trna_b4")
    objs_trna_a = find_mesh("trna_d4")
    pep_search = os.path.splitext(os.path.basename(peptide_pdb))[0]
    objs_pep = find_mesh(pep_search)

    print(f"  Found: surface={[o.name for o in objs_surface]}, "
          f"mRNA={[o.name for o in objs_mrna]}, "
          f"tRNA_P={[o.name for o in objs_trna_p]}, "
          f"tRNA_A={[o.name for o in objs_trna_a]}, "
          f"peptide={[o.name for o in objs_pep]}")

    if not all([objs_surface, objs_mrna, objs_trna_p, objs_trna_a, objs_pep]):
        print(f"  ERROR: Missing objects")
        return

    obj_surface = objs_surface[0]
    obj_trna_p = objs_trna_p[0]
    obj_trna_a = objs_trna_a[0]
    obj_mrna = objs_mrna[0]
    obj_peptide = objs_pep[0]

    # Apply z-rotation to all primary objects (matches animate.py orientation)
    for o in [obj_surface, obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]:
        o.rotation_euler.z = math.pi / 2

    # --- Apply mRNA bend (organic curvature outside ribosome) ---
    mrna_mesh_res_ids = get_mesh_res_ids(obj_mrna)
    n_mrna_verts = len(obj_mrna.data.vertices)
    mrna_co = np.empty(n_mrna_verts * 3, dtype=np.float64)
    obj_mrna.data.vertices.foreach_get('co', mrna_co)
    mrna_positions = mrna_co.reshape(-1, 3).copy()
    print(f"  Applying mRNA bend (channel +/-{MRNA_CHANNEL_HALF_LEN} BU, "
          f"strength {MRNA_BEND_STRENGTH})...")
    mrna_positions = apply_mrna_bend(mrna_positions, mrna_mesh_res_ids)
    obj_mrna.data.vertices.foreach_set('co', mrna_positions.ravel())
    obj_mrna.data.update()

    # --- Camera setup ---
    # Use exact camera angle from Blender (aligned via View > Align Active
    # Camera to View). Rotation and target from scene.blend, pushed back
    # for ortho rendering.
    import mathutils
    canvas.frame_object(mol_surface)
    cam = scene.camera
    print(f"  Camera (auto): loc={tuple(cam.location)}, lens={cam.data.lens}")

    # Rotation from scene.blend (View > Align Active Camera to View)
    cam_rot = mathutils.Euler((2.2480, 0.0, 0.0489), 'XYZ')
    # Orbit center from Blender viewport
    target = mathutils.Vector((-2.66, 1.71, 1.72))
    # Camera forward = local -Z rotated by cam_rot
    forward = mathutils.Vector((0, 0, -1))
    forward.rotate(cam_rot)
    # Push camera 50 BU back from target (ortho — distance doesn't affect size)
    cam.location = target - forward * 50.0
    cam.rotation_euler = cam_rot

    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 9.0    # BU visible — ribosome fills frame
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0
    print(f"  Camera (Blender-matched ortho): loc={tuple(cam.location)}, "
          f"rot={tuple(cam.rotation_euler)}, ortho_scale={cam.data.ortho_scale}")

    bpy.context.view_layer.update()

    # --- Save .blend or render ---
    if SAVE_BLEND:
        blend_path = os.path.abspath(BLEND_FILE)
        print(f"  Saving scene to {blend_path}...")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        size_mb = os.path.getsize(blend_path) / (1024 * 1024)
        print(f"  Saved: {blend_path} ({size_mb:.1f} MB)")
        print("=== Done (scene saved, no render) ===")
        return

    # --- Two-pass render + composite (outline ribosome) ---
    import time
    internal_objs = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]

    # Pass 1: Ribosome silhouette (transparent bg → alpha = shape mask)
    print("  Pass 1: Ribosome silhouette...")
    t0 = time.time()
    for o in internal_objs:
        o.hide_render = True
    scene.render.film_transparent = True
    canvas.snapshot("renders/_pass_ribo.png")
    t1 = time.time()
    print(f"    [{t1 - t0:.1f}s]")

    # Pass 2: Internal components (no ribosome)
    print("  Pass 2: Internal components...")
    obj_surface.hide_render = True
    for o in internal_objs:
        o.hide_render = False
    scene.render.film_transparent = False
    canvas.snapshot("renders/_pass_internal.png")
    t2 = time.time()
    print(f"    [{t2 - t1:.1f}s]")

    # Composite: edge-detect ribosome alpha → outline overlay
    print("  Compositing...")
    OUTLINE_COLOR = (70, 120, 200)
    OUTLINE_THICKNESS = 3

    internal = Image.open("renders/_pass_internal.png").convert("RGBA")
    ribo = Image.open("renders/_pass_ribo.png").convert("RGBA")

    # Extract alpha channel → binary mask → edge detect
    alpha = np.array(ribo)[:, :, 3]
    mask = (alpha > 10).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=2))
    mask_img = Image.fromarray((np.array(mask_img) > 128).astype(np.uint8) * 255)
    edges = mask_img.filter(ImageFilter.FIND_EDGES)
    sil = Image.fromarray((np.array(edges) > 30).astype(np.uint8) * 255)
    for _ in range(OUTLINE_THICKNESS // 2):
        sil = sil.filter(ImageFilter.MaxFilter(3))

    # Build outline overlay
    edges_np = np.array(sil)
    overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
    mask = edges_np > 100
    overlay[mask, 0] = OUTLINE_COLOR[0]
    overlay[mask, 1] = OUTLINE_COLOR[1]
    overlay[mask, 2] = OUTLINE_COLOR[2]
    overlay[mask, 3] = 255

    result = Image.alpha_composite(internal, Image.fromarray(overlay, "RGBA"))
    result.save(OUTPUT_FILE)
    t3 = time.time()
    print(f"    [{t3 - t2:.1f}s]")
    print(f"  Saved: {OUTPUT_FILE}")
    print(f"=== Done (total render: {t3 - t0:.1f}s) ===")


if __name__ == "__main__":
    main()
