"""Animate 10 elongation cycles of the 6Y0G human 80S ribosome.

v3: Visual overhaul
- Ribosome jitter (subtle rigid-body motion)
- mRNA: no rigid-body rotation, PCA deformation + reduced per-atom jitter
- tRNA tumbling during approach/departure
- Polypeptide: no rigid-body jitter, no choreographic motion, just progressive reveal
- PCA structural modes for physically realistic deformation

Renders N_CYCLES * FRAMES_PER_CYCLE frames (seamless loop) showing:
- tRNA delivery to A-site (with tumbling)
- Peptide transfer with progressive polypeptide reveal
- Translocation (A->P, P->E)
- tRNA departure (with tumbling)

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
JITTER_HARMONICS = 4

# Target frequencies (cycles/frame) matching original visual feel.
# Rounded to nearest integer harmonic of TOTAL_FRAMES for seamless looping.
_TARGET_FREQS = [0.07, 0.113, 0.183, 0.296]
JITTER_HARMONIC_NUMBERS = [max(1, round(f * TOTAL_FRAMES)) for f in _TARGET_FREQS]
JITTER_FREQS = [k / TOTAL_FRAMES for k in JITTER_HARMONIC_NUMBERS]

# Ribosome jitter (subtle: issue 1)
RIBO_JITTER_TRANS_AMP = 0.03  # BU per axis
RIBO_JITTER_ROT_AMP = 1.5     # degrees per axis

# tRNA jitter (moderate)
TRNA_JITTER_TRANS_AMP = 0.12  # BU per axis
TRNA_JITTER_ROT_AMP = 8.0     # degrees per axis

# Per-atom thermal jitter amplitudes (per object type)
ATOM_JITTER_MRNA = 0.1        # BU — reduced from 0.3 (issue 3)
ATOM_JITTER_TRNA = 0.2        # BU
ATOM_JITTER_PEPTIDE = 0.15    # BU

ATOM_SPATIAL_VECS = np.array([
    [0.13, 0.17, 0.11],
    [0.19, -0.11, 0.15],
    [-0.14, 0.18, 0.12],
    [0.16, 0.13, -0.17],
])

# PCA deformation amplitude (BU, scaled per mode)
PCA_BASE_AMP = 0.5  # base amplitude for mode 0, decreases for higher modes


def compute_jitter(global_frame, obj_index, trans_amp, rot_amp):
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
        trans[axis] = trans_amp * t_sum / norm
        rot[axis] = math.radians(rot_amp) * r_sum / norm
    return trans, rot


def compute_per_atom_jitter(global_frame, obj_index, positions, amplitude):
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
    return amplitude * noise / norm


# ---------------------------------------------------------------------------
# PCA deformation
# ---------------------------------------------------------------------------
def load_pca_modes(npz_path):
    """Load PCA modes from npz file. Returns (modes, residue_ids) or (None, None)."""
    if not os.path.exists(npz_path):
        print(f"  WARNING: {npz_path} not found, PCA deformation disabled")
        return None, None
    data = np.load(npz_path)
    modes = data['modes']  # (n_modes, n_residues, 3)
    residue_ids = data['residue_ids']
    print(f"  Loaded {npz_path}: {modes.shape[0]} modes, {modes.shape[1]} residues")
    return modes, residue_ids


def compute_pca_displacement(global_frame, modes, obj_index):
    """Compute per-residue PCA displacement for a given frame.

    Returns (n_residues, 3) displacement vector.
    Uses integer-harmonic sines with different frequency per mode.
    """
    if modes is None:
        return None

    n_modes, n_res, _ = modes.shape
    displacement = np.zeros((n_res, 3))

    for m in range(n_modes):
        # Each mode gets a unique harmonic number (avoid collisions with jitter)
        harmonic_num = max(1, JITTER_HARMONIC_NUMBERS[0] + m * 3 + obj_index * 2)
        freq = harmonic_num / TOTAL_FRAMES
        phase = (m * 3.7 + obj_index * 5.1) % (2 * math.pi)
        amplitude = PCA_BASE_AMP / (m + 1)  # decreasing amplitude for higher modes

        val = math.sin(2 * math.pi * freq * global_frame + phase)
        displacement += amplitude * val * modes[m]

    return displacement


def apply_pca_to_vertices(positions, res_ids, pca_displacement, pca_residue_ids):
    """Apply per-residue PCA displacement to mesh vertices.

    Maps PCA residue indices to mesh res_id attribute.
    """
    if pca_displacement is None or res_ids is None:
        return positions

    result = positions.copy()
    unique_mesh_res = np.unique(res_ids)

    # Map PCA residue indices to mesh residues (by order)
    n_pca = len(pca_residue_ids)
    n_mesh = len(unique_mesh_res)
    n_map = min(n_pca, n_mesh)

    for i in range(n_map):
        mesh_res = unique_mesh_res[i]
        mask = res_ids == mesh_res
        # PCA displacement is in Angstroms, mesh is in BU (1 BU = 10 A)
        result[mask] += pca_displacement[i] * 0.1  # A -> BU

    return result


# ---------------------------------------------------------------------------
# tRNA tumbling
# ---------------------------------------------------------------------------
def compute_tumble_rotation(global_frame, obj_index, max_angle_deg=180):
    """Compute tumbling rotation angles for a tRNA in solution.

    Returns (rx, ry, rz) Euler angles.
    Uses integer-harmonic sines with unique frequencies per axis.
    """
    rot = np.zeros(3)
    for axis in range(3):
        total = 0.0
        for h in range(JITTER_HARMONICS):
            harmonic_num = max(1, JITTER_HARMONIC_NUMBERS[h] + axis * 5 + obj_index * 11)
            freq = harmonic_num / TOTAL_FRAMES
            phase = (obj_index * 11.3 + axis * 4.1 + h * 2.3) % (2 * math.pi)
            total += math.sin(2 * math.pi * freq * global_frame + phase) / (h + 1)
        norm = sum(1.0 / (k + 1) for k in range(JITTER_HARMONICS))
        rot[axis] = math.radians(max_angle_deg) * total / norm
    return rot


# ---------------------------------------------------------------------------
# Animation position calculator (single-cycle logic)
# ---------------------------------------------------------------------------
def get_positions(local_frame):
    """Return (mRNA_delta, tRNA_P_delta, tRNA_A_delta) for one cycle.

    local_frame: frame index within a single cycle (0 to FRAMES_PER_CYCLE-1).
    All deltas are relative to the object's crystallographic pose (0 = no movement).
    Does NOT include cumulative offsets — caller adds those.

    Note: polypeptide no longer has choreographic motion (issue 8).
    """
    f0 = scale_frames(0)
    f30 = scale_frames(30)
    f90 = scale_frames(90)
    f120 = scale_frames(120)
    f150 = scale_frames(150)
    f210 = scale_frames(210)
    f240 = scale_frames(240)

    zero = np.zeros(3)

    # --- mRNA (translation only, no rotation — issue 3) ---
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

    return mrna_delta, trna_p_delta, trna_a_delta


def get_trna_tumble_factor(local_frame, site):
    """Return tumble amplitude factor [0, 1] for tRNA.

    A-site tRNA:
      - Full tumble during approach (before accommodation at f90)
      - Ramps down during accommodation (f30-f90)
      - Zero when bound

    P-site tRNA:
      - Zero when bound
      - Ramps up during departure (f210-f240)
    """
    f30 = scale_frames(30)
    f90 = scale_frames(90)
    f210 = scale_frames(210)
    f240 = scale_frames(240)

    if site == "A":
        if local_frame < f30:
            return 1.0  # full tumble during approach
        elif local_frame < f90:
            # Decay during accommodation
            t = frame_t(local_frame, f30, f90)
            return 1.0 - t
        else:
            return 0.0  # bound, no tumble
    elif site == "P":
        if local_frame < f210:
            return 0.0  # bound, no tumble
        elif local_frame < f240:
            # Ramp up during departure
            t = frame_t(local_frame, f210, f240)
            return t
        else:
            return 1.0  # fully departed


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

    # --- Load PCA modes ---
    mrna_modes, mrna_pca_res = load_pca_modes("mrna_modes.npz")
    trna_modes, trna_pca_res = load_pca_modes("trna_modes.npz")

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

    # 5. Polypeptide (tunnel-threaded from preprocessed PDB — issue 9)
    peptide_pdb = "tunnel_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        print(f"  WARNING: {peptide_pdb} not found, falling back to extended_polypeptide.pdb")
        peptide_pdb = "extended_polypeptide.pdb"
    mol_peptide = mn.Molecule.load(peptide_pdb)
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
    # Match mRNA/peptide objects by PDB filename
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

    # --- Get mesh res_ids for PCA mapping ---
    mrna_mesh_res_ids = get_mesh_res_ids(obj_mrna)
    trna_p_mesh_res_ids = get_mesh_res_ids(obj_trna_p)
    trna_a_mesh_res_ids = get_mesh_res_ids(obj_trna_a)

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

            # Single-cycle deltas (no polypeptide choreography — issue 8)
            mrna_d, trna_p_d, trna_a_d = get_positions(local_frame)

            # Cumulative mRNA offset (slides further each cycle)
            mrna_d = mrna_d + cycle * CODON_SHIFT

            # --- Ribosome jitter (issue 1) ---
            ribo_t, ribo_r = compute_jitter(
                global_frame, 10, RIBO_JITTER_TRANS_AMP, RIBO_JITTER_ROT_AMP)
            obj_surface.location = tuple(ribo_t)
            obj_surface.rotation_euler = (ribo_r[0], ribo_r[1],
                                          math.pi / 2 + ribo_r[2])

            # --- mRNA: translation only, NO rigid-body rotation (issue 3) ---
            obj_mrna.location = tuple(mrna_d)
            obj_mrna.rotation_euler = (0, 0, math.pi / 2)

            # --- tRNA jitter + tumbling ---
            # P-site tRNA
            trna_p_jitter_t, trna_p_jitter_r = compute_jitter(
                global_frame, 1, TRNA_JITTER_TRANS_AMP, TRNA_JITTER_ROT_AMP)

            tumble_factor_p = get_trna_tumble_factor(local_frame, "P")
            if tumble_factor_p > 0:
                tumble_p = compute_tumble_rotation(global_frame, 1)
                trna_p_jitter_r = trna_p_jitter_r + tumble_factor_p * np.array(tumble_p)

            obj_trna_p.location = tuple(trna_p_d + trna_p_jitter_t)
            obj_trna_p.rotation_euler = (trna_p_jitter_r[0], trna_p_jitter_r[1],
                                         math.pi / 2 + trna_p_jitter_r[2])

            # A-site tRNA (issue 6: tumbling during approach)
            trna_a_jitter_t, trna_a_jitter_r = compute_jitter(
                global_frame, 2, TRNA_JITTER_TRANS_AMP, TRNA_JITTER_ROT_AMP)

            tumble_factor_a = get_trna_tumble_factor(local_frame, "A")
            if tumble_factor_a > 0:
                tumble_a = compute_tumble_rotation(global_frame, 2)
                trna_a_jitter_r = trna_a_jitter_r + tumble_factor_a * np.array(tumble_a)

            obj_trna_a.location = tuple(trna_a_d + trna_a_jitter_t)
            obj_trna_a.rotation_euler = (trna_a_jitter_r[0], trna_a_jitter_r[1],
                                         math.pi / 2 + trna_a_jitter_r[2])

            # --- Polypeptide: NO rigid-body jitter, NO choreographic motion (issue 8) ---
            obj_peptide.location = (0, 0, 0)
            obj_peptide.rotation_euler = (0, 0, math.pi / 2)

            # --- Per-atom deformation ---
            for obj_idx, obj in enumerate(jitter_objects):
                orig = orig_verts[obj.name]

                if obj is obj_peptide:
                    # Polypeptide: progressive reveal + small per-atom jitter (issue 8)
                    revealed = compute_peptide_positions(
                        orig, pep_res_ids, cycle, local_frame)
                    displacement = compute_per_atom_jitter(
                        global_frame, obj_idx, orig, ATOM_JITTER_PEPTIDE)
                    displaced = (revealed + displacement).ravel()

                elif obj is obj_mrna:
                    # mRNA: PCA deformation + reduced per-atom jitter (issues 3, 4, 7)
                    positions = orig.copy()
                    pca_disp = compute_pca_displacement(
                        global_frame, mrna_modes, obj_index=0)
                    positions = apply_pca_to_vertices(
                        positions, mrna_mesh_res_ids, pca_disp, mrna_pca_res)
                    displacement = compute_per_atom_jitter(
                        global_frame, obj_idx, orig, ATOM_JITTER_MRNA)
                    displaced = (positions + displacement).ravel()

                elif obj is obj_trna_p:
                    # P-site tRNA: PCA deformation + per-atom jitter
                    positions = orig.copy()
                    pca_disp = compute_pca_displacement(
                        global_frame, trna_modes, obj_index=1)
                    positions = apply_pca_to_vertices(
                        positions, trna_p_mesh_res_ids, pca_disp, trna_pca_res)
                    displacement = compute_per_atom_jitter(
                        global_frame, obj_idx, orig, ATOM_JITTER_TRNA)
                    displaced = (positions + displacement).ravel()

                elif obj is obj_trna_a:
                    # A-site tRNA: PCA deformation + per-atom jitter
                    positions = orig.copy()
                    pca_disp = compute_pca_displacement(
                        global_frame, trna_modes, obj_index=2)
                    positions = apply_pca_to_vertices(
                        positions, trna_a_mesh_res_ids, pca_disp, trna_pca_res)
                    displacement = compute_per_atom_jitter(
                        global_frame, obj_idx, orig, ATOM_JITTER_TRNA)
                    displaced = (positions + displacement).ravel()

                else:
                    displacement = compute_per_atom_jitter(
                        global_frame, obj_idx, orig, ATOM_JITTER_TRNA)
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
