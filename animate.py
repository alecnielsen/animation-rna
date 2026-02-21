"""Animate 10 elongation cycles of the 6Y0G human 80S ribosome.

Renders N_CYCLES * FRAMES_PER_CYCLE frames (seamless loop) showing:
- tRNA delivery to A-site
- Peptide transfer with progressive polypeptide reveal
- Translocation (A->P, P->E)
- tRNA departure

Seamless loop via conveyor belt: mRNA extends far beyond camera frame,
absorbing cumulative codon shifts. Polypeptide grows 1 residue per cycle.

Two-pass rendering per frame (internal cartoon + surface), saved to renders/frames/.

Run with: python3.11 animate.py [--debug]
  --debug: 480x270, 24 frames/cycle, 4 samples (fast preview)
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
N_CYCLES = 10

if DEBUG:
    RES = (480, 270)
    FRAMES_PER_CYCLE = 24
    SAMPLES_INTERNAL = 4
    SAMPLES_SURFACE = 4
else:
    RES = (1920, 1080)
    FRAMES_PER_CYCLE = 240
    SAMPLES_INTERNAL = 48
    SAMPLES_SURFACE = 16

TOTAL_FRAMES = N_CYCLES * FRAMES_PER_CYCLE
FPS = 24

if DEBUG:
    print(f"=== DEBUG MODE: {RES[0]}x{RES[1]}, {N_CYCLES} cycles x "
          f"{FRAMES_PER_CYCLE} frames = {TOTAL_FRAMES} total, "
          f"{SAMPLES_INTERNAL} samples ===")

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
# ---------------------------------------------------------------------------
PA_VEC = np.array((-2.51, 1.86, 0.05))       # P-site to A-site direction
EP_VEC = -PA_VEC                               # A-site to P-site = E to P direction
CODON_SHIFT = np.array((-0.75, 0.35, -0.56))  # one codon mRNA advance

ENTRY_OFFSET = 3.0 * PA_VEC   # starting position for incoming tRNA
DEPART_OFFSET = 3.0 * EP_VEC  # departure position for leaving tRNA

RIBO_CENTROID = np.array((-23.90, 24.24, 22.56))
CAMERA_ORBIT_DEGREES = 0  # disabled while iterating

# Polypeptide progressive reveal
INITIAL_PEPTIDE_RESIDUES = 2  # visible at start (matching C4 dipeptide)


# ---------------------------------------------------------------------------
# Material helpers
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
# Keyframe interpolation helpers
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
    """Scale frame number from 240-frame schedule to actual FRAMES_PER_CYCLE."""
    return int(round(f * FRAMES_PER_CYCLE / 240))


# ---------------------------------------------------------------------------
# Molecular jitter — sum-of-sines with integer-harmonic frequencies
#
# Frequencies are integer multiples of 1/TOTAL_FRAMES, guaranteeing that
# sin(2pi * freq * TOTAL_FRAMES + phase) == sin(phase) — perfect loop.
# ---------------------------------------------------------------------------
JITTER_TRANS_AMP = 0.15   # BU per axis (rigid-body translation)
JITTER_ROT_AMP = 15.0     # degrees per axis (rigid-body rotation)
JITTER_HARMONICS = 4

# Target frequencies (cycles/frame) matching original visual feel.
# Rounded to nearest integer harmonic of TOTAL_FRAMES for seamless looping.
_TARGET_FREQS = [0.07, 0.113, 0.183, 0.296]
JITTER_HARMONIC_NUMBERS = [max(1, round(f * TOTAL_FRAMES)) for f in _TARGET_FREQS]
JITTER_FREQS = [k / TOTAL_FRAMES for k in JITTER_HARMONIC_NUMBERS]

# Per-atom thermal jitter
ATOM_JITTER_AMP = 0.3     # BU per atom displacement amplitude
ATOM_SPATIAL_VECS = np.array([
    [0.13, 0.17, 0.11],
    [0.19, -0.11, 0.15],
    [-0.14, 0.18, 0.12],
    [0.16, 0.13, -0.17],
])


def compute_jitter(global_frame, obj_index):
    """Return (trans_xyz, rot_xyz) rigid-body jitter for given frame and object index."""
    trans = np.zeros(3)
    rot = np.zeros(3)
    for axis in range(3):
        t_sum = 0.0
        r_sum = 0.0
        for h in range(JITTER_HARMONICS):
            phase = (obj_index * 7.3 + axis * 2.9 + h * 1.7) % (2 * math.pi)
            val = math.sin(2 * math.pi * JITTER_FREQS[h] * global_frame + phase)
            t_sum += val / (h + 1)
            r_sum += val / (h + 1)
        norm = sum(1.0 / (h + 1) for h in range(JITTER_HARMONICS))
        trans[axis] = JITTER_TRANS_AMP * t_sum / norm
        rot[axis] = math.radians(JITTER_ROT_AMP) * r_sum / norm
    return trans, rot


def compute_per_atom_jitter(global_frame, obj_index, positions):
    """Return (N, 3) per-atom displacement with spatial correlation."""
    n = len(positions)
    noise = np.zeros((n, 3))
    for h in range(JITTER_HARMONICS):
        freq = JITTER_FREQS[h]
        time_val = 2 * math.pi * freq * global_frame
        spatial_phase = positions @ ATOM_SPATIAL_VECS[h]
        for axis in range(3):
            phase_offset = axis * 2.9 + obj_index * 7.3 + h * 1.7
            noise[:, axis] += np.sin(time_val + spatial_phase + phase_offset) / (h + 1)
    norm = sum(1.0 / (k + 1) for k in range(JITTER_HARMONICS))
    return ATOM_JITTER_AMP * noise / norm


# ---------------------------------------------------------------------------
# Animation position calculator (single-cycle logic)
# ---------------------------------------------------------------------------
def get_positions(local_frame):
    """Return (mRNA_delta, tRNA_P_delta, tRNA_A_delta, peptide_delta) for one cycle.

    local_frame: frame index within a single cycle (0 to FRAMES_PER_CYCLE-1).
    All deltas are relative to the object's crystallographic pose (0 = no movement).
    Does NOT include cumulative offsets — caller adds those.
    """
    f0 = scale_frames(0)
    f30 = scale_frames(30)
    f90 = scale_frames(90)
    f120 = scale_frames(120)
    f150 = scale_frames(150)
    f210 = scale_frames(210)
    f240 = scale_frames(240)

    zero = np.zeros(3)

    # --- mRNA ---
    if local_frame < f150:
        mrna_delta = zero.copy()
    elif local_frame < f210:
        t = frame_t(local_frame, f150, f210)
        mrna_delta = lerp(zero, CODON_SHIFT, t)
    else:
        mrna_delta = CODON_SHIFT.copy()

    # --- P-site tRNA ---
    if local_frame < f150:
        trna_p_delta = zero.copy()
    elif local_frame < f210:
        t = frame_t(local_frame, f150, f210)
        trna_p_delta = lerp(zero, EP_VEC, t)
    elif local_frame < f240:
        t = frame_t(local_frame, f210, f240)
        trna_p_delta = lerp(EP_VEC, EP_VEC + DEPART_OFFSET, t)
    else:
        trna_p_delta = EP_VEC + DEPART_OFFSET

    # --- A-site tRNA ---
    if local_frame < f30:
        trna_a_delta = PA_VEC + ENTRY_OFFSET
    elif local_frame < f90:
        t = frame_t(local_frame, f30, f90)
        trna_a_delta = lerp(PA_VEC + ENTRY_OFFSET, PA_VEC, t)
    elif local_frame < f150:
        trna_a_delta = PA_VEC.copy()
    elif local_frame < f210:
        t = frame_t(local_frame, f150, f210)
        trna_a_delta = lerp(PA_VEC, zero, t)
    else:
        trna_a_delta = zero.copy()

    # --- Polypeptide ---
    if local_frame < f120:
        peptide_delta = zero.copy()
    elif local_frame < f150:
        t = frame_t(local_frame, f120, f150)
        peptide_delta = lerp(zero, PA_VEC, t)
    elif local_frame < f210:
        t = frame_t(local_frame, f150, f210)
        peptide_delta = lerp(PA_VEC, zero, t)
    else:
        peptide_delta = zero.copy()

    return mrna_delta, trna_p_delta, trna_a_delta, peptide_delta


# ---------------------------------------------------------------------------
# Polypeptide progressive reveal
# ---------------------------------------------------------------------------
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


def compute_peptide_positions(orig_positions, res_ids, cycle, local_frame):
    """Apply progressive reveal: collapse unrevealed residues to anchor point.

    Returns modified vertex positions with unrevealed residues collapsed to the
    last visible residue's centroid. During peptide transfer phase, the next
    residue smoothly interpolates from collapsed to true position.
    """
    positions = orig_positions.copy()
    unique_res = np.sort(np.unique(res_ids))
    n_res = len(unique_res)

    # Number of fully visible residues at start of this cycle
    base_visible = min(INITIAL_PEPTIDE_RESIDUES + cycle, n_res)

    # During peptide transfer (f120-f150), animate next residue appearing
    f120 = scale_frames(120)
    f150 = scale_frames(150)

    if f120 <= local_frame < f150:
        reveal_t = (local_frame - f120) / max(f150 - f120, 1)
    elif local_frame >= f150:
        reveal_t = 1.0
    else:
        reveal_t = 0.0

    fully_visible = min(base_visible + (1 if reveal_t >= 1.0 else 0), n_res)
    animating_idx = base_visible if reveal_t > 0 and reveal_t < 1.0 else -1

    if fully_visible >= n_res:
        return positions  # all residues visible

    # Anchor: centroid of last fully visible residue
    anchor_res = unique_res[fully_visible - 1]
    anchor = orig_positions[res_ids == anchor_res].mean(axis=0)

    for i in range(n_res):
        if i < fully_visible:
            continue  # keep original positions

        res_mask = res_ids == unique_res[i]

        if i == animating_idx and 0 < reveal_t < 1:
            # Smoothly interpolate from anchor to true position
            positions[res_mask] = anchor + reveal_t * (orig_positions[res_mask] - anchor)
        else:
            # Collapse to anchor
            positions[res_mask] = anchor

    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mn.register()

    print(f"=== Loading scene ({N_CYCLES} cycles x {FRAMES_PER_CYCLE} = "
          f"{TOTAL_FRAMES} frames @ {FPS}fps) ===")
    print(f"  Jitter harmonics: {JITTER_HARMONIC_NUMBERS} "
          f"(freqs: {[f'{f:.4f}' for f in JITTER_FREQS]})")

    canvas = mn.Canvas(mn.scene.Cycles(samples=SAMPLES_INTERNAL), resolution=RES)
    scene = bpy.context.scene
    scene.render.film_transparent = False
    set_bg(scene, (0.04, 0.04, 0.06), 0.5)
    scene.cycles.max_bounces = 12

    # --- Load molecules ---
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

    # 4. A-site tRNA (chain B4)
    mol_trna_a = mn.Molecule.fetch("6Y0G")
    mol_trna_a.add_style(
        style=mn.StyleCartoon(),
        selection=mol_trna_a.select.chain_id(["B4"]),
        material=make_solid_material((0.95, 0.5, 0.1)),
        name="tRNA_A",
    )

    # 5. Polypeptide (extended helix from preprocessed PDB)
    mol_peptide = mn.Molecule.load("extended_polypeptide.pdb")
    mol_peptide.add_style(
        style=mn.StyleCartoon(),
        material=make_solid_material((0.8, 0.15, 0.6)),
        name="polypeptide",
    )

    # --- Find Blender objects ---
    objs_6y0g = sorted(
        [o for o in bpy.data.objects if "6Y0G" in o.name and o.type == "MESH"],
        key=lambda o: o.name,
    )
    objs_mrna = [o for o in bpy.data.objects if "extended_mrna" in o.name and o.type == "MESH"]
    objs_pep = [o for o in bpy.data.objects if "extended_polypeptide" in o.name and o.type == "MESH"]

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

    # Apply rotation to all primary objects
    primary_objects = [obj_surface, obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    for o in primary_objects:
        o.rotation_euler.z = math.pi / 2

    animated_objects = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]

    # --- Store original vertex positions for per-atom jitter ---
    jitter_objects = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    orig_verts = {}
    for obj in jitter_objects:
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', co)
        orig_verts[obj.name] = co.reshape(-1, 3).copy()
        print(f"  Stored {n} vertices for {obj.name}")

    # --- Polypeptide residue mapping for progressive reveal ---
    pep_res_ids = get_mesh_res_ids(obj_peptide)
    if pep_res_ids is not None:
        pep_n_res = len(np.unique(pep_res_ids))
        print(f"  Polypeptide: {pep_n_res} residues from mesh attribute")
    else:
        # Fallback: assume 5 atoms per ALA residue (N, CA, C, O, CB)
        n_verts = len(obj_peptide.data.vertices)
        atoms_per_res = 5
        n_res_est = n_verts // atoms_per_res
        pep_res_ids = np.repeat(np.arange(1, n_res_est + 1), atoms_per_res)[:n_verts]
        pep_n_res = n_res_est
        print(f"  Polypeptide: {pep_n_res} residues (estimated, {atoms_per_res} atoms/res)")

    # --- Camera setup ---
    canvas.frame_object(mol_surface)
    cam = scene.camera
    cam_loc_orig = np.array(cam.location)
    print(f"  Camera: loc={tuple(cam_loc_orig)}, lens={cam.data.lens}")

    bpy.ops.object.empty_add(type="PLAIN_AXES", location=tuple(RIBO_CENTROID))
    orbit_empty = bpy.context.active_object
    orbit_empty.name = "CameraOrbit"
    cam.parent = orbit_empty
    cam.matrix_parent_inverse = orbit_empty.matrix_world.inverted()

    # --- Render loop ---
    print(f"\n=== Rendering {TOTAL_FRAMES} frames ({N_CYCLES} cycles) ===")

    for cycle in range(N_CYCLES):
        for local_frame in range(FRAMES_PER_CYCLE):
            global_frame = cycle * FRAMES_PER_CYCLE + local_frame
            print(f"\n--- Cycle {cycle}, Frame {local_frame}/{FRAMES_PER_CYCLE - 1} "
                  f"(global {global_frame}/{TOTAL_FRAMES - 1}) ---")

            # Single-cycle deltas
            mrna_d, trna_p_d, trna_a_d, peptide_d = get_positions(local_frame)

            # Cumulative mRNA offset (slides further each cycle)
            mrna_d = mrna_d + cycle * CODON_SHIFT

            # Rigid-body jitter (uses global frame for seamless looping)
            jitter_mrna_t, jitter_mrna_r = compute_jitter(global_frame, 0)
            jitter_trna_p_t, jitter_trna_p_r = compute_jitter(global_frame, 1)
            jitter_trna_a_t, jitter_trna_a_r = compute_jitter(global_frame, 2)
            jitter_pep_t, jitter_pep_r = compute_jitter(global_frame, 3)

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

            # Per-atom jitter for non-peptide objects
            for obj_idx, obj in enumerate(jitter_objects):
                orig = orig_verts[obj.name]
                if obj is obj_peptide:
                    # Polypeptide: apply progressive reveal then jitter
                    revealed = compute_peptide_positions(
                        orig, pep_res_ids, cycle, local_frame)
                    displacement = compute_per_atom_jitter(global_frame, obj_idx, orig)
                    displaced = (revealed + displacement).ravel()
                else:
                    displacement = compute_per_atom_jitter(global_frame, obj_idx, orig)
                    displaced = (orig + displacement).ravel()
                obj.data.vertices.foreach_set('co', displaced)
                obj.data.update()

            # Camera orbit
            orbit_t = global_frame / max(TOTAL_FRAMES - 1, 1)
            orbit_angle = math.radians(CAMERA_ORBIT_DEGREES) * orbit_t
            orbit_empty.rotation_euler.z = orbit_angle

            bpy.context.view_layer.update()

            # --- Pass 1: Internal components (cartoon) ---
            obj_surface.hide_render = True
            for o in animated_objects:
                o.hide_render = False

            set_bg(scene, (0.04, 0.04, 0.06), 0.5)
            scene.render.film_transparent = False
            scene.cycles.samples = SAMPLES_INTERNAL

            pass1_path = os.path.join(FRAMES_DIR, f"pass1_{global_frame:04d}.png")
            canvas.snapshot(pass1_path)
            print(f"  Pass 1 saved: {pass1_path}")

            # --- Pass 2: Surface (ribosome only) ---
            obj_surface.hide_render = False
            for o in animated_objects:
                o.hide_render = True

            set_bg(scene, (0.02, 0.02, 0.03), 0.3)
            scene.render.film_transparent = True
            scene.cycles.samples = SAMPLES_SURFACE

            pass2_path = os.path.join(FRAMES_DIR, f"pass2_{global_frame:04d}.png")
            canvas.snapshot(pass2_path)
            print(f"  Pass 2 saved: {pass2_path}")

    print(f"\n=== Done! {TOTAL_FRAMES} frames rendered to {FRAMES_DIR}/ ===")
    print("Next: python3.11 composite.py [--debug]")


if __name__ == "__main__":
    main()
