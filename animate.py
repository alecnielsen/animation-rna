"""Animate one elongation cycle of the 6Y0G human 80S ribosome.

Renders 240 frames (24fps = 10 seconds) showing:
- tRNA delivery to A-site
- Peptide transfer
- Translocation (A→P, P→E)
- tRNA departure

Two-pass rendering per frame (internal cartoon + surface), saved to renders/frames/.

Run with: python3.11 animate.py [--debug]
  --debug: 480x270, 24 frames, 4 samples (fast preview)
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
    RES = (480, 270)
    TOTAL_FRAMES = 24
    SAMPLES_INTERNAL = 4
    SAMPLES_SURFACE = 4
    print("=== DEBUG MODE: 480x270, 24 frames, 4 samples ===")
else:
    RES = (1920, 1080)
    TOTAL_FRAMES = 240
    SAMPLES_INTERNAL = 48
    SAMPLES_SURFACE = 16

FPS = 24

# Output dirs
FRAMES_DIR = "renders/frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Chain definitions (from render.py)
# ---------------------------------------------------------------------------
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
# Position constants (Blender units / nm, from measure_positions.py)
# These are absolute centroid positions within the mesh geometry.
# Since MN objects start at origin, animation deltas move the object.
# ---------------------------------------------------------------------------
# Delta vectors for animation (object-space movement)
PA_VEC = np.array((-2.51, 1.86, 0.05))       # P-site to A-site direction
EP_VEC = -PA_VEC                               # A-site to P-site = E to P direction
CODON_SHIFT = np.array((-0.75, 0.35, -0.56))  # one codon mRNA advance

# Entry position offset: 2x PA_VEC beyond A-site (well outside ribosome)
ENTRY_OFFSET = 3.0 * PA_VEC   # starting position for incoming tRNA (relative to A-site)
# E-site departure offset: 2x beyond E-site
DEPART_OFFSET = 3.0 * EP_VEC  # departure position for leaving tRNA

# Ribosome visual centroid (from measure_positions.py) — camera orbits around this
RIBO_CENTROID = np.array((-23.90, 24.24, 22.56))

# Camera orbit
CAMERA_ORBIT_DEGREES = 0  # disabled while iterating on other aspects


# ---------------------------------------------------------------------------
# Material helpers (from render.py)
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


# ---------------------------------------------------------------------------
# Keyframe interpolation helper
# ---------------------------------------------------------------------------
def lerp(a, b, t):
    """Linear interpolation between vectors a and b, t in [0, 1]."""
    return a + (b - a) * np.clip(t, 0, 1)


def frame_t(frame, start, end):
    """Return normalized time [0,1] for frame within [start, end] range."""
    if frame < start:
        return 0.0
    if frame >= end:
        return 1.0
    return (frame - start) / (end - start)


def scale_frames(f):
    """Scale frame number from 240-frame schedule to actual TOTAL_FRAMES."""
    return int(round(f * TOTAL_FRAMES / 240))


# ---------------------------------------------------------------------------
# Molecular jitter — sum-of-sines with irrational frequency ratios
# ---------------------------------------------------------------------------
JITTER_TRANS_AMP = 0.15   # BU per axis (rigid-body translation)
JITTER_ROT_AMP = 15.0     # degrees per axis (rigid-body rotation)
JITTER_HARMONICS = 4
# Golden-ratio-spaced frequencies (irrational → never repeats)
PHI = (1 + math.sqrt(5)) / 2
JITTER_FREQS = [0.07 * PHI**k for k in range(JITTER_HARMONICS)]

# Per-atom thermal jitter
ATOM_JITTER_AMP = 0.3     # BU per atom — displacement amplitude
# Spatial frequency vectors: each harmonic uses a different direction so
# the noise has complex spatial structure (not just a single plane wave).
ATOM_SPATIAL_VECS = np.array([
    [0.13, 0.17, 0.11],
    [0.19, -0.11, 0.15],
    [-0.14, 0.18, 0.12],
    [0.16, 0.13, -0.17],
])


def compute_jitter(frame, obj_index):
    """Return (trans_xyz, rot_xyz) rigid-body jitter for given frame and object index."""
    trans = np.zeros(3)
    rot = np.zeros(3)
    for axis in range(3):
        t_sum = 0.0
        r_sum = 0.0
        for h in range(JITTER_HARMONICS):
            phase = (obj_index * 7.3 + axis * 2.9 + h * 1.7) % (2 * math.pi)
            val = math.sin(2 * math.pi * JITTER_FREQS[h] * frame + phase)
            t_sum += val / (h + 1)
            r_sum += val / (h + 1)
        norm = sum(1.0 / (h + 1) for h in range(JITTER_HARMONICS))
        trans[axis] = JITTER_TRANS_AMP * t_sum / norm
        rot[axis] = math.radians(JITTER_ROT_AMP) * r_sum / norm
    return trans, rot


def compute_per_atom_jitter(frame, obj_index, positions):
    """Return (N, 3) per-atom displacement with spatial correlation.

    Nearby atoms move similarly (correlated via position-dependent phase),
    giving a natural thermal wobble rather than dissolution. Each harmonic
    uses a different spatial direction so the deformation is complex, not
    a single plane wave.
    """
    n = len(positions)
    noise = np.zeros((n, 3))
    for h in range(JITTER_HARMONICS):
        freq = JITTER_FREQS[h]
        time_val = 2 * math.pi * freq * frame
        spatial_phase = positions @ ATOM_SPATIAL_VECS[h]  # (n,) — varies smoothly with position
        for axis in range(3):
            phase_offset = axis * 2.9 + obj_index * 7.3 + h * 1.7
            noise[:, axis] += np.sin(time_val + spatial_phase + phase_offset) / (h + 1)
    norm = sum(1.0 / (k + 1) for k in range(JITTER_HARMONICS))
    return ATOM_JITTER_AMP * noise / norm


# ---------------------------------------------------------------------------
# Animation position calculator
# ---------------------------------------------------------------------------
def get_positions(frame):
    """Return (mRNA_delta, tRNA_P_delta, tRNA_A_delta, peptide_delta) for a given frame.

    All deltas are relative to the object's original position (i.e., 0 = crystallographic pose).
    The tRNA_P object uses chain B4 (P-site tRNA in the crystal structure).
    The tRNA_A object also uses chain B4 geometry, but we pre-offset it to the A-site.

    In the crystal structure:
    - B4 is at the P-site
    - D4 is at the A-site
    We use B4 geometry for both tRNA objects. tRNA_A starts pre-displaced by PA_VEC
    to overlay with D4's position, then gets additional animation offsets on top.
    """
    # Scale frame thresholds from 240 to TOTAL_FRAMES
    f0 = scale_frames(0)
    f30 = scale_frames(30)
    f90 = scale_frames(90)
    f120 = scale_frames(120)
    f150 = scale_frames(150)
    f210 = scale_frames(210)
    f240 = scale_frames(240)

    zero = np.zeros(3)

    # --- mRNA ---
    # Static until translocation (f150-f210), then slides one codon
    if frame < f150:
        mrna_delta = zero.copy()
    elif frame < f210:
        t = frame_t(frame, f150, f210)
        mrna_delta = lerp(zero, CODON_SHIFT, t)
    else:
        mrna_delta = CODON_SHIFT.copy()

    # --- P-site tRNA (starts at P-site = origin for B4 object) ---
    # Static until translocation, then moves P→E, then departs
    if frame < f150:
        trna_p_delta = zero.copy()
    elif frame < f210:
        # P → E site
        t = frame_t(frame, f150, f210)
        trna_p_delta = lerp(zero, EP_VEC, t)
    elif frame < f240:
        # Depart from E-site
        t = frame_t(frame, f210, f240)
        trna_p_delta = lerp(EP_VEC, EP_VEC + DEPART_OFFSET, t)
    else:
        trna_p_delta = EP_VEC + DEPART_OFFSET

    # --- A-site tRNA (B4 geometry, pre-offset by ENTRY_OFFSET+PA_VEC to start outside) ---
    # Relative to the A-site position (which is PA_VEC from P-site / B4 origin)
    # Entry animation moves from far outside → A-site
    # Then translocation moves A-site → P-site
    if frame < f30:
        # Waiting outside: at entry position
        trna_a_delta = PA_VEC + ENTRY_OFFSET
    elif frame < f90:
        # Glide from entry to A-site
        t = frame_t(frame, f30, f90)
        trna_a_delta = lerp(PA_VEC + ENTRY_OFFSET, PA_VEC, t)
    elif frame < f150:
        # Settled at A-site
        trna_a_delta = PA_VEC.copy()
    elif frame < f210:
        # Translocation: A → P
        t = frame_t(frame, f150, f210)
        trna_a_delta = lerp(PA_VEC, zero, t)
    else:
        # At P-site (new P-site tRNA)
        trna_a_delta = zero.copy()

    # --- Polypeptide ---
    # Follows P-site tRNA until peptide transfer, then jumps to A-site tRNA
    if frame < f120:
        # At P-site (following P-site tRNA, which is static)
        peptide_delta = zero.copy()
    elif frame < f150:
        # Peptide transfer: jump from P-site to A-site
        t = frame_t(frame, f120, f150)
        peptide_delta = lerp(zero, PA_VEC, t)
    elif frame < f210:
        # Translocation: follows A-site tRNA back to P-site
        t = frame_t(frame, f150, f210)
        peptide_delta = lerp(PA_VEC, zero, t)
    else:
        peptide_delta = zero.copy()

    return mrna_delta, trna_p_delta, trna_a_delta, peptide_delta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mn.register()

    print(f"=== Loading scene ({TOTAL_FRAMES} frames @ {FPS}fps) ===")

    # Create initial canvas for scene setup
    canvas = mn.Canvas(mn.scene.Cycles(samples=SAMPLES_INTERNAL), resolution=RES)
    scene = bpy.context.scene
    scene.render.film_transparent = False
    set_bg(scene, (0.04, 0.04, 0.06), 0.5)
    scene.cycles.max_bounces = 12

    # --- Load molecules (4 fetched from 6Y0G + 1 extended mRNA from PDB) ---
    print("  Loading molecules...")

    # 1. Ribosome surface (40S + 60S)
    mol_surface = mn.Molecule.fetch("6Y0G")
    mol_surface.add_style(
        style=mn.StyleSurface(),
        selection=mol_surface.select.chain_id(RIBOSOME_CHAINS),
        material=make_surface_material(),
        name="surface",
    )

    # 2. mRNA (extended, from preprocessed PDB)
    mol_mrna = mn.Molecule.load("extended_mrna.pdb")
    mol_mrna.add_style(
        style=mn.StyleCartoon(),
        material=make_solid_material((0.1, 0.35, 0.95)),
        name="mRNA",
    )

    # 3. P-site tRNA (chain B4)
    mol_trna_p = mn.Molecule.fetch("6Y0G")
    mol_trna_p.add_style(
        style=mn.StyleCartoon(),
        selection=mol_trna_p.select.chain_id(["B4"]),
        material=make_solid_material((0.95, 0.5, 0.1)),
        name="tRNA_P",
    )

    # 4. A-site tRNA (chain B4 — same geometry, different object)
    mol_trna_a = mn.Molecule.fetch("6Y0G")
    mol_trna_a.add_style(
        style=mn.StyleCartoon(),
        selection=mol_trna_a.select.chain_id(["B4"]),
        material=make_solid_material((0.95, 0.5, 0.1)),
        name="tRNA_A",
    )

    # 5. Polypeptide (chain C4)
    mol_peptide = mn.Molecule.fetch("6Y0G")
    mol_peptide.add_style(
        style=mn.StyleCartoon(),
        selection=mol_peptide.select.chain_id(["C4"]),
        material=make_solid_material((0.8, 0.15, 0.6)),
        name="polypeptide",
    )

    # --- Find Blender objects ---
    # 4 fetched 6Y0G objects + 1 loaded extended_mrna
    objs_6y0g = sorted(
        [o for o in bpy.data.objects if "6Y0G" in o.name and o.type == "MESH"],
        key=lambda o: o.name,
    )
    objs_mrna = [o for o in bpy.data.objects if "extended_mrna" in o.name and o.type == "MESH"]
    print(f"  Found {len(objs_6y0g)} 6Y0G objects: {[o.name for o in objs_6y0g]}")
    print(f"  Found {len(objs_mrna)} mRNA objects: {[o.name for o in objs_mrna]}")

    if len(objs_6y0g) < 4 or len(objs_mrna) < 1:
        print(f"  ERROR: Expected 4 6Y0G + 1 mRNA objects")
        return

    obj_surface = objs_6y0g[0]
    obj_trna_p = objs_6y0g[1]
    obj_trna_a = objs_6y0g[2]
    obj_peptide = objs_6y0g[3]
    obj_mrna = objs_mrna[0]

    # Apply rotation to primary objects (same as render.py)
    primary_objects = [obj_surface, obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    for o in primary_objects:
        o.rotation_euler.z = math.pi / 2

    animated_objects = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    all_objects = [obj_surface] + animated_objects

    # --- Store original vertex positions for per-atom jitter ---
    jitter_objects = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    orig_verts = {}
    for obj in jitter_objects:
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', co)
        orig_verts[obj.name] = co.reshape(-1, 3).copy()
        print(f"  Stored {n} vertices for {obj.name}")

    # --- Camera setup ---
    # Frame on ribosome surface first
    canvas.frame_object(mol_surface)
    cam = scene.camera
    cam_loc_orig = np.array(cam.location)
    cam_rot_orig = np.array(cam.rotation_euler)
    cam_lens = cam.data.lens
    cam_clip = (cam.data.clip_start, cam.data.clip_end)
    print(f"  Camera: loc={tuple(cam_loc_orig)}, lens={cam_lens}")

    # Create empty at ribosome centroid for camera orbit
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(RIBO_CENTROID))
    orbit_empty = bpy.context.active_object
    orbit_empty.name = "CameraOrbit"

    # Parent camera to orbit empty
    cam.parent = orbit_empty
    # Keep camera's world transform by adjusting local transform
    cam.matrix_parent_inverse = orbit_empty.matrix_world.inverted()

    # --- Render loop ---
    print(f"\n=== Rendering {TOTAL_FRAMES} frames ===")

    for frame in range(TOTAL_FRAMES):
        print(f"\n--- Frame {frame}/{TOTAL_FRAMES - 1} ---")

        # Compute animation positions
        mrna_d, trna_p_d, trna_a_d, peptide_d = get_positions(frame)

        # Compute jitter
        jitter_mrna_t, jitter_mrna_r = compute_jitter(frame, 0)
        jitter_trna_p_t, jitter_trna_p_r = compute_jitter(frame, 1)
        jitter_trna_a_t, jitter_trna_a_r = compute_jitter(frame, 2)
        jitter_pep_t, jitter_pep_r = compute_jitter(frame, 3)

        # Apply position deltas + jitter
        obj_mrna.location = tuple(mrna_d + jitter_mrna_t)
        obj_mrna.rotation_euler = (jitter_mrna_r[0], jitter_mrna_r[1],
                                   math.pi / 2 + jitter_mrna_r[2])

        obj_trna_p.location = tuple(trna_p_d + jitter_trna_p_t)
        obj_trna_p.rotation_euler = (jitter_trna_p_r[0], jitter_trna_p_r[1],
                                     math.pi / 2 + jitter_trna_p_r[2])

        obj_trna_a.location = tuple(trna_a_d + jitter_trna_a_t)
        obj_trna_a.rotation_euler = (jitter_trna_a_r[0], jitter_trna_a_r[1],
                                     math.pi / 2 + jitter_trna_a_r[2])

        obj_peptide.location = tuple(peptide_d + jitter_pep_t)
        obj_peptide.rotation_euler = (jitter_pep_r[0], jitter_pep_r[1],
                                      math.pi / 2 + jitter_pep_r[2])

        # Per-atom thermal jitter (displace mesh vertices)
        for obj_idx, obj in enumerate(jitter_objects):
            orig = orig_verts[obj.name]
            displacement = compute_per_atom_jitter(frame, obj_idx, orig)
            displaced = (orig + displacement).ravel()
            obj.data.vertices.foreach_set('co', displaced)
            obj.data.update()

        # Camera orbit
        orbit_t = frame / max(TOTAL_FRAMES - 1, 1)
        orbit_angle = math.radians(CAMERA_ORBIT_DEGREES) * orbit_t
        orbit_empty.rotation_euler.z = orbit_angle

        # Force scene update
        bpy.context.view_layer.update()

        # --- Pass 1: Internal components (cartoon) ---
        obj_surface.hide_render = True
        for o in animated_objects:
            o.hide_render = False

        set_bg(scene, (0.04, 0.04, 0.06), 0.5)
        scene.render.film_transparent = False
        scene.cycles.samples = SAMPLES_INTERNAL

        pass1_path = os.path.join(FRAMES_DIR, f"pass1_{frame:04d}.png")
        canvas.snapshot(pass1_path)
        print(f"  Pass 1 saved: {pass1_path}")

        # --- Pass 2: Surface (ribosome only) ---
        obj_surface.hide_render = False
        for o in animated_objects:
            o.hide_render = True

        set_bg(scene, (0.02, 0.02, 0.03), 0.3)
        scene.render.film_transparent = True
        scene.cycles.samples = SAMPLES_SURFACE

        pass2_path = os.path.join(FRAMES_DIR, f"pass2_{frame:04d}.png")
        canvas.snapshot(pass2_path)
        print(f"  Pass 2 saved: {pass2_path}")

    print(f"\n=== Done! {TOTAL_FRAMES} frames rendered to {FRAMES_DIR}/ ===")
    print("Next: python3.11 composite.py [--debug]")


if __name__ == "__main__":
    main()
