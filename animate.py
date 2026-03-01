"""Animate seamless-looping ribosome translation with repeating folded domains.

v8: Polypeptide anchoring + GLY-aware folding.
- Gradient scroll: tunnel residues anchored at PTC, smooth ramp to full scroll
  for external residues (fixes unfolding-back-into-ribosome and tRNA detachment)
- GLY-aware folded residue ranges (4 atoms for GLY vs 5 for others)
- Extended tail from 10→40 residues for complete distal polypeptide

v7: Physics-based thermal motion via per-frame OpenMM MD.
- Replaces sinusoidal jitter/PCA with Langevin dynamics (310K, position restraints)
- mRNA stationary at origin (tRNA choreography implies codon reading)
- 3 mobile molecules (mRNA, tRNA-P, tRNA-A): per-frame CPU MD (k=100)
- Polypeptide: folding morph animation (no MD — backbone-only PDB)
- Ribosome: pre-computed ENM trajectory from Modal GPU (ribosome_thermal.npz)
- Cross-fade last 12 frames for seamless loop blending
- Polypeptide morph coords converted Å→BU to match mesh vertex space

v6: Repeating HP35 domain polypeptide with folding morph animation.
- 38 cycles (one repeat unit = 35 res domain + 3 res linker) for seamless loop
- 12 frames/cycle at 24fps = 2 AA/s elongation rate (debug: 6 frames/cycle)
- Domain folding: extended→folded morph with N-to-C wave propagation
- Scrolling: polypeptide scrolls by repeat_distance/38 per cycle
- Per-domain fold scheduling: nearest-to-tunnel folds, others fully folded

6-phase per-cycle choreography (from v5):
  Phase 1: ESTABLISH        f0-f12    (5%)
  Phase 2: tRNA DELIVERY    f12-f96   (35%)
  Phase 3: ACCOMMODATION    f96-f120  (10%)
  Phase 4: PEPTIDE TRANSFER f120-f144 (10%)
  Phase 5: TRANSLOCATION    f144-f192 (20%)
  Phase 6: tRNA DEPARTURE   f192-f240 (20%)

Renders N_CYCLES * FRAMES_PER_CYCLE frames (seamless loop).

Run with: python3.11 animate.py [--debug]
  --debug: 480x270, 6 frames/cycle, 32 samples (fast preview)
"""

import molecularnodes as mn
import bpy
import numpy as np
import os
import sys
import math
from PIL import Image, ImageFilter

ANG_TO_BU = 0.01  # MolecularNodes world_scale: 1 Å = 0.01 BU

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEBUG = "--debug" in sys.argv
SAVE_BLEND = "--save-blend" in sys.argv
N_CYCLES = 38  # one repeat unit (35 res domain + 3 res linker)

# Molecule style: cartoon (fast) or surface (production)
MOL_STYLE = "cartoon"
for arg in sys.argv:
    if arg.startswith("--style="):
        MOL_STYLE = arg.split("=", 1)[1]

if DEBUG:
    RES = (480, 270)
    FRAMES_PER_CYCLE = 6   # debug: 6 frames/cycle -> 228 frames total
    SAMPLES = 32
else:
    RES = (1920, 1080)
    FRAMES_PER_CYCLE = 12  # production: 12 frames/cycle @ 24fps = 2 AA/s
    SAMPLES = 64

TOTAL_FRAMES = N_CYCLES * FRAMES_PER_CYCLE
FPS = 24

# Fold data (loaded from NPZ in main())
FOLD_DATA = None
SCROLL_PER_CYCLE = None  # computed from NPZ: repeat_distance / N_CYCLES
SCROLL_VECTOR = None     # unit vector along chain exit direction

if DEBUG:
    print(f"=== DEBUG MODE: {RES[0]}x{RES[1]}, {N_CYCLES} cycles x "
          f"{FRAMES_PER_CYCLE} frames = {TOTAL_FRAMES} total, "
          f"{SAMPLES} samples ===")

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

# Polypeptide (legacy progressive reveal kept for fallback)
INITIAL_PEPTIDE_RESIDUES = 200  # visible at start (long chain extending out of tunnel)

# mRNA bend — droop outside the ribosome channel
MRNA_CHANNEL_HALF_LEN = 4.0   # BU — straight zone around mRNA centroid
MRNA_BEND_STRENGTH = 0.015    # BU per BU² beyond channel (quadratic droop)


# ---------------------------------------------------------------------------
# mRNA bend (organic curvature outside ribosome)
# ---------------------------------------------------------------------------
def apply_mrna_bend(positions, res_ids):
    """Apply a gentle quadratic droop to mRNA vertices outside the ribosome channel.

    Vertices within ±MRNA_CHANNEL_HALF_LEN of the mRNA centroid along the
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
    # Use cross product with Z to get a horizontal perpendicular
    z_up = np.array([0.0, 0.0, 1.0])
    droop_dir = np.cross(local_axis, z_up)
    droop_norm = np.linalg.norm(droop_dir)
    if droop_norm < 1e-6:
        # local_axis is nearly vertical, use X instead
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
            # Sign: both ends droop in the same direction (gravity-like)
            droop = MRNA_BEND_STRENGTH * d * d
            positions[i] += droop * droop_dir

    return positions


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


def make_translucent_surface_material(style="hashed"):
    """Translucent material for the ribosome.

    Styles:
      hashed  — Principled BSDF Alpha=0.12 + HASHED blend (denoiser smooths noise)
      volume  — Volume Absorption shader (physically-based, thin=transparent)
      sss     — Principled BSDF with Subsurface Scattering (waxy translucent)
    """
    mat = bpy.data.materials.new(name="translucent_surface")
    n = mat.node_tree.nodes
    l = mat.node_tree.links
    n.clear()
    out = n.new("ShaderNodeOutputMaterial")

    if style == "volume":
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
        bsdf = n.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = (0.45, 0.55, 0.75, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.4
        bsdf.inputs["Subsurface Weight"].default_value = 0.8
        bsdf.inputs["Subsurface Radius"].default_value = (0.5, 0.5, 0.5)
        bsdf.inputs["Subsurface Scale"].default_value = 0.5
        l.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    else:  # "hashed" (default)
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


def apply_md_deltas_to_mesh(positions, res_ids, md_deltas, md_residue_ids):
    """Apply per-residue MD displacement to mesh vertices.

    Maps MD residue indices to mesh res_id attribute by ordinal position.
    md_deltas: (n_residues, 3) in Blender units.
    """
    if md_deltas is None or res_ids is None:
        return positions

    result = positions.copy()
    unique_mesh_res = np.unique(res_ids)

    n_md = len(md_residue_ids)
    n_mesh = len(unique_mesh_res)
    n_map = min(n_md, n_mesh)

    for i in range(n_map):
        mesh_res = unique_mesh_res[i]
        mask = res_ids == mesh_res
        result[mask] += md_deltas[i]

    return result


# ---------------------------------------------------------------------------
# Per-frame OpenMM MD thermal motion
# ---------------------------------------------------------------------------
LOOP_BLEND_FRAMES = 12  # cross-fade last N frames for seamless loop

class MolecularDynamics:
    """Per-frame OpenMM MD simulation for physics-based thermal motion.

    Runs a continuous Langevin dynamics trajectory with position restraints
    (keeps atoms near rest positions) and wall repulsion (prevents clipping
    through ribosome). Each call to step_and_get_deltas() advances the
    simulation and returns per-residue centroid displacements in BU.
    """

    def __init__(self, label, pdb_path, ribo_coords_A,
                 k_restraint=100.0, k_wall=1000.0, temperature=310,
                 steps_per_frame=500, mol_type="rna"):
        import tempfile
        from openmm.app import (
            PDBFile as OmmPDB, ForceField, Modeller, Simulation,
            CutoffNonPeriodic, HBonds,
        )
        from openmm import (
            LangevinMiddleIntegrator, CustomExternalForce, Platform,
        )
        from openmm.unit import kelvin, picosecond, picoseconds, nanometer
        from scipy.spatial import KDTree

        self.label = label
        self.steps_per_frame = steps_per_frame
        self._early_deltas = []  # store first LOOP_BLEND_FRAMES for cross-fade

        print(f"  MD init: {label} ({mol_type}, k={k_restraint}, T={temperature}K)")

        # Strip CONECT records
        with open(pdb_path) as f:
            lines = [line for line in f if not line.startswith("CONECT")]
        clean = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
        clean.writelines(lines)
        clean.close()

        try:
            pdb = OmmPDB(clean.name)
        finally:
            os.unlink(clean.name)

        ff = ForceField("amber14-all.xml")

        modeller = Modeller(pdb.topology, pdb.positions)

        # RNA 5' terminus: remove OP3/P/OP1/OP2 so AMBER14 recognizes
        # the 5' end (crystal structures have OP3, tiled mRNA has P)
        if mol_type == "rna":
            first_res = list(modeller.topology.residues())[0]
            to_remove = [a for a in first_res.atoms()
                         if a.name in ("P", "OP1", "OP2", "OP3")]
            if to_remove:
                modeller.delete(to_remove)
            modeller.addHydrogens(ff)

        # Protein: add hydrogens (terminal residues handled automatically)
        if mol_type == "protein":
            modeller.addHydrogens(ff)

        n_atoms = modeller.topology.getNumAtoms()
        print(f"    {n_atoms} atoms (with H)")

        # Create system: in-vacuo, CutoffNonPeriodic for fast pair computation
        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=CutoffNonPeriodic,
            nonbondedCutoff=1.0 * nanometer,
            constraints=HBonds,
        )

        # Position restraints: E = 0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)
        restraint = CustomExternalForce(
            "0.5*k_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k_restraint", k_restraint)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")

        positions = modeller.positions
        for i in range(n_atoms):
            pos = positions[i].value_in_unit(nanometer)
            restraint.addParticle(i, [pos[0], pos[1], pos[2]])
        system.addForce(restraint)

        # Wall repulsion from ribosome atoms
        if ribo_coords_A is not None and len(ribo_coords_A) > 0:
            ribo_tree = KDTree(ribo_coords_A)
            mol_coords_nm = np.array([positions[i].value_in_unit(nanometer)
                                       for i in range(n_atoms)])
            mol_coords_A = mol_coords_nm * 10.0

            _, nearest_idx = ribo_tree.query(mol_coords_A)
            nearest_ribo_nm = ribo_coords_A[nearest_idx] * 0.1

            wall_force = CustomExternalForce(
                "0.5*k_wall*step(r_min-dist)*((r_min-dist)^2);"
                "dist=sqrt((x-wx)^2+(y-wy)^2+(z-wz)^2);"
                "r_min=0.3"
            )
            wall_force.addGlobalParameter("k_wall", k_wall)
            wall_force.addPerParticleParameter("wx")
            wall_force.addPerParticleParameter("wy")
            wall_force.addPerParticleParameter("wz")

            for i in range(n_atoms):
                wall_force.addParticle(i, [
                    nearest_ribo_nm[i, 0], nearest_ribo_nm[i, 1],
                    nearest_ribo_nm[i, 2],
                ])
            system.addForce(wall_force)

        # Integrator + simulation (CPU platform)
        integrator = LangevinMiddleIntegrator(
            temperature * kelvin, 1 / picosecond, 0.002 * picoseconds)
        platform = Platform.getPlatformByName('CPU')
        self.sim = Simulation(modeller.topology, system, integrator, platform)
        self.sim.context.setPositions(modeller.positions)

        # Minimize
        print(f"    Minimizing (500 iterations)...")
        self.sim.minimizeEnergy(maxIterations=500)

        # Thermalize (burn-in)
        print(f"    Thermalizing (1000 steps)...")
        self.sim.step(1000)

        # Build residue → atom index mapping
        self._residue_atoms = {}
        for residue in self.sim.topology.residues():
            self._residue_atoms[residue.index] = [a.index for a in residue.atoms()]
        self.residue_ids = np.array(sorted(self._residue_atoms.keys()))
        self.n_residues = len(self.residue_ids)

        # Record rest centroids (post-thermalization)
        from openmm.unit import angstrom
        state = self.sim.context.getState(getPositions=True)
        pos_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        self._rest_centroids = np.zeros((self.n_residues, 3))
        for ri, res_id in enumerate(self.residue_ids):
            atom_idx = self._residue_atoms[res_id]
            self._rest_centroids[ri] = pos_A[atom_idx].mean(axis=0)

        print(f"    Ready: {self.n_residues} residues, "
              f"{steps_per_frame} steps/frame")

    def step_and_get_deltas(self, global_frame):
        """Run MD steps and return per-residue centroid deltas in BU.

        Returns (n_residues, 3) displacement from rest position.
        Handles seamless loop blending for last LOOP_BLEND_FRAMES frames.
        """
        from openmm.unit import angstrom

        try:
            self.sim.step(self.steps_per_frame)
        except Exception as e:
            print(f"    WARNING: {self.label} MD step failed ({e}), returning zeros")
            return np.zeros((self.n_residues, 3))

        state = self.sim.context.getState(getPositions=True)
        pos_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)

        centroids = np.zeros((self.n_residues, 3))
        for ri, res_id in enumerate(self.residue_ids):
            atom_idx = self._residue_atoms[res_id]
            centroids[ri] = pos_A[atom_idx].mean(axis=0)

        # Deltas in Angstroms, convert to BU (MN world_scale = 0.01)
        deltas_BU = (centroids - self._rest_centroids) * ANG_TO_BU

        # Store early frames for cross-fade
        if len(self._early_deltas) < LOOP_BLEND_FRAMES:
            self._early_deltas.append(deltas_BU.copy())

        return deltas_BU

    def get_blended_deltas(self, global_frame, total_frames, raw_deltas):
        """Cross-fade last LOOP_BLEND_FRAMES back to early frames for seamless loop."""
        frames_from_end = total_frames - 1 - global_frame
        if frames_from_end < LOOP_BLEND_FRAMES and len(self._early_deltas) > 0:
            blend_idx = LOOP_BLEND_FRAMES - 1 - frames_from_end
            if blend_idx < len(self._early_deltas):
                t = (frames_from_end + 1) / LOOP_BLEND_FRAMES  # 1→0 as we approach end
                t = t * t * (3.0 - 2.0 * t)  # smoothstep
                return t * raw_deltas + (1.0 - t) * self._early_deltas[blend_idx]
        return raw_deltas


# ---------------------------------------------------------------------------
# Animation position calculator (single-cycle logic)
#
# v4 schedule (in 240-frame units):
#   Phase 1: ESTABLISH        f0-f12    (5%)
#   Phase 2: tRNA DELIVERY    f12-f96   (35%)
#   Phase 3: ACCOMMODATION    f96-f120  (10%)
#   Phase 4: PEPTIDE TRANSFER f120-f144 (10%)
#   Phase 5: TRANSLOCATION    f144-f192 (20%)
#   Phase 6: tRNA DEPARTURE   f192-f240 (20%)
# ---------------------------------------------------------------------------
def get_positions(local_frame):
    """Return (mRNA_delta, tRNA_P_delta, tRNA_A_delta) for one cycle.

    local_frame: frame index within a single cycle (0 to FRAMES_PER_CYCLE-1).
    All deltas are relative to the object's crystallographic pose (0 = no movement).
    Does NOT include cumulative offsets -- caller adds those.
    """
    f0 = scale_frames(0)
    f12 = scale_frames(12)
    f96 = scale_frames(96)
    f120 = scale_frames(120)
    f144 = scale_frames(144)
    f192 = scale_frames(192)
    f240 = scale_frames(240)

    zero = np.zeros(3)

    # --- mRNA (translation only, no rotation) ---
    if local_frame < f144:
        mrna_delta = zero.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        mrna_delta = lerp(zero, CODON_SHIFT, t)
    else:
        mrna_delta = CODON_SHIFT.copy()

    # --- P-site tRNA ---
    if local_frame < f144:
        trna_p_delta = zero.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        trna_p_delta = lerp(zero, EP_VEC, t)
    elif local_frame < f240:
        t = frame_t(local_frame, f192, f240)
        trna_p_delta = lerp(EP_VEC, EP_VEC + DEPART_OFFSET, t)
    else:
        trna_p_delta = EP_VEC + DEPART_OFFSET

    # --- A-site tRNA ---
    if local_frame < f12:
        trna_a_delta = PA_VEC + ENTRY_OFFSET
    elif local_frame < f96:
        t = frame_t(local_frame, f12, f96)
        trna_a_delta = lerp(PA_VEC + ENTRY_OFFSET, PA_VEC, t)
    elif local_frame < f144:
        trna_a_delta = PA_VEC.copy()
    elif local_frame < f192:
        t = frame_t(local_frame, f144, f192)
        trna_a_delta = lerp(PA_VEC, zero, t)
    else:
        trna_a_delta = zero.copy()

    return mrna_delta, trna_p_delta, trna_a_delta


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

    # During peptide transfer (f120-f144), animate next residue appearing
    f120 = scale_frames(120)
    f144 = scale_frames(144)

    if f120 <= local_frame < f144:
        reveal_t = (local_frame - f120) / max(f144 - f120, 1)
    elif local_frame >= f144:
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
# Polypeptide folding morph (v6: repeating HP35 domains)
# ---------------------------------------------------------------------------
def smoothstep(t):
    """Hermite smoothstep: t^2 * (3 - 2*t). Starts slow, accelerates, settles."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _build_folded_residue_ranges(fold_data, domain_idx):
    """Build per-residue atom ranges for a folded domain using atom_names.

    HP35 contains glycines (4 atoms: N,CA,C,O — no CB), so the simple
    `ri * 5` indexing is wrong.  Instead, detect residue boundaries from
    the atom_names array where 'N' marks the start of each residue.

    Returns: list of (start, end) index tuples into the folded coord array.
    """
    key = f'domain_{domain_idx}_atom_names'
    if key not in fold_data:
        # Fallback: assume 5 atoms per residue (all ALA)
        n = len(fold_data[f'domain_{domain_idx}_folded'])
        n_res = n // 5
        return [(i * 5, min(i * 5 + 5, n)) for i in range(n_res)]

    atom_names = fold_data[key]
    ranges = []
    cur_start = 0
    for j in range(1, len(atom_names)):
        if atom_names[j] == 'N':
            ranges.append((cur_start, j))
            cur_start = j
    ranges.append((cur_start, len(atom_names)))
    return ranges


def compute_polypeptide_morph(orig_positions, res_ids, fold_data,
                               cycle, local_frame, frames_per_cycle):
    """Per-frame polypeptide vertex positions with folding + scrolling.

    1. Compute global progress (fractional cycles elapsed)
    2. For each domain: determine fold_t, interpolate extended<->folded
    3. Apply gradient scroll: tunnel residues anchored at PTC, external
       residues scroll at full rate, smooth ramp in between

    Args:
        orig_positions: (N, 3) original vertex positions (extended conformation)
        res_ids: (N,) per-vertex residue IDs
        fold_data: dict from NPZ with domain coords and metadata
        cycle: current cycle index (0 to N_CYCLES-1)
        local_frame: frame within cycle (0 to frames_per_cycle-1)
        frames_per_cycle: frames per cycle

    Returns: (N, 3) modified vertex positions
    """
    positions = orig_positions.copy()
    n_domains = int(fold_data['n_domains'])
    scroll_vector = fold_data['scroll_vector']
    repeat_distance = float(fold_data['repeat_distance'])
    scroll_per_cycle = repeat_distance / N_CYCLES

    # Global progress in fractional cycles
    global_progress = cycle + local_frame / frames_per_cycle

    # --- Tunnel exit residue: infer from domain 0 start ---
    # Tunnel residues are numbered 1..(domain_0_start - 1).
    # They should stay anchored at the PTC (no scroll).
    tunnel_exit_res = int(fold_data['domain_0_start_res']) - 1
    # Transition zone: ramp scroll factor from 0→1 over a few residues
    # around the tunnel exit so the chain doesn't have a hard kink.
    SCROLL_RAMP_RESIDUES = 8

    for di in range(n_domains):
        start_res = int(fold_data[f'domain_{di}_start_res'])
        end_res = int(fold_data[f'domain_{di}_end_res'])
        folded_coords = fold_data[f'domain_{di}_folded'] * ANG_TO_BU    # Å → BU
        extended_coords = fold_data[f'domain_{di}_extended'] * ANG_TO_BU  # Å → BU

        # Domain mask: vertices belonging to this domain
        domain_mask = (res_ids >= start_res) & (res_ids <= end_res)
        if not np.any(domain_mask):
            continue

        # Domain fold scheduling:
        # fold_t: 0 = extended, 1 = folded (mesh starts folded from PDB)
        # - Domain 0 (nearest tunnel exit): fold_t ramps 0->1 over 38 cycles
        # - All other domains: fold_t = 1.0 (fully folded, no morph needed)
        if di == 0:
            fold_t = np.clip(global_progress / N_CYCLES, 0.0, 1.0)
        else:
            fold_t = 1.0

        if fold_t >= 1.0:
            continue  # fully folded, mesh already in correct position

        # Get domain vertex indices
        domain_indices = np.where(domain_mask)[0]

        # Map mesh vertices to domain atom coordinates by residue ID
        domain_res_unique = np.unique(res_ids[domain_mask])
        n_domain_atoms = len(folded_coords)
        n_domain_verts = len(domain_indices)

        # Build proper per-residue atom ranges (handles GLY with 4 atoms)
        folded_res_ranges = _build_folded_residue_ranges(fold_data, di)
        # Extended always has 5 atoms/res (all ALA backbone)
        n_ext_atoms = len(extended_coords)

        if n_domain_verts != n_domain_atoms:
            # MN mesh may have different vertex count than PDB atoms
            # Fall back to per-residue centroid morphing
            for ri, res in enumerate(domain_res_unique):
                res_mask = domain_mask & (res_ids == res)
                if not np.any(res_mask):
                    continue

                # Per-residue fold_t with N-to-C wave
                n_res = len(domain_res_unique)
                residue_frac = ri / max(n_res - 1, 1)
                per_res_t = np.clip((fold_t - residue_frac * 0.5) / 0.5, 0.0, 1.0)
                eased_t = smoothstep(per_res_t)

                # Mesh is loaded in FOLDED conformation from the PDB.
                # unfold_t=1 → fully extended, unfold_t=0 → stay folded.
                unfold_t = 1.0 - eased_t
                if unfold_t <= 0.0:
                    continue  # fully folded, no displacement needed

                # Folded centroid: use proper atom ranges (handles GLY)
                if ri < len(folded_res_ranges):
                    f_start, f_end = folded_res_ranges[ri]
                    folded_centroid = folded_coords[f_start:f_end].mean(axis=0)
                else:
                    continue

                # Extended centroid: always 5 atoms/res
                ext_start = ri * 5
                ext_end = min(ext_start + 5, n_ext_atoms)
                if ext_start >= n_ext_atoms:
                    continue
                extended_centroid = extended_coords[ext_start:ext_end].mean(axis=0)

                # Unfold displacement: move from folded mesh toward extended
                displacement = unfold_t * (extended_centroid - folded_centroid)
                positions[res_mask] += displacement
        else:
            # Direct 1:1 mapping between mesh vertices and domain atoms
            for vi, idx in enumerate(domain_indices):
                res = res_ids[idx]
                ri = np.searchsorted(domain_res_unique, res)
                n_res = len(domain_res_unique)
                residue_frac = ri / max(n_res - 1, 1)
                per_res_t = np.clip((fold_t - residue_frac * 0.5) / 0.5, 0.0, 1.0)
                eased_t = smoothstep(per_res_t)
                unfold_t = 1.0 - eased_t

                if unfold_t > 0.0 and vi < n_domain_atoms:
                    # Mesh starts folded; displace toward extended
                    delta = unfold_t * (extended_coords[vi] - folded_coords[vi])
                    positions[idx] = orig_positions[idx] + delta

    # --- Gradient scroll: anchor tunnel at PTC, full scroll for external ---
    # Scroll offset at full rate (what external residues get)
    scroll_offset = global_progress * scroll_per_cycle * scroll_vector * ANG_TO_BU

    # Compute per-vertex scroll factor based on residue position
    unique_res = np.sort(np.unique(res_ids))
    scroll_factors = np.zeros(len(positions))
    for res in unique_res:
        mask = res_ids == res
        if res <= tunnel_exit_res - SCROLL_RAMP_RESIDUES:
            # Deep in tunnel: no scroll (anchored at PTC/tRNA)
            factor = 0.0
        elif res <= tunnel_exit_res:
            # Transition zone: smooth ramp from 0 to ~0.5
            t = (res - (tunnel_exit_res - SCROLL_RAMP_RESIDUES)) / SCROLL_RAMP_RESIDUES
            factor = smoothstep(t) * 0.5
        elif res <= tunnel_exit_res + SCROLL_RAMP_RESIDUES:
            # Just past exit: ramp from 0.5 to 1.0
            t = (res - tunnel_exit_res) / SCROLL_RAMP_RESIDUES
            factor = 0.5 + smoothstep(t) * 0.5
        else:
            # External chain: full scroll
            factor = 1.0
        scroll_factors[mask] = factor

    positions += scroll_factors[:, np.newaxis] * scroll_offset

    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write_backbone(in_pdb, out_pdb, mol_type="rna"):
    """Write a backbone-only PDB for cleaner visualization."""
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
    """Extract tRNA chains from 6Y0G as separate PDB files if not already cached."""
    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile as BiotitePDB
    import biotite.structure.io.pdbx as pdbx
    from pathlib import Path

    chains_to_extract = {"B4": "trna_b4.pdb", "D4": "trna_d4.pdb"}
    if all(os.path.exists(f) for f in chains_to_extract.values()):
        print("  tRNA PDBs already cached")
        return

    cache_dir = Path.home() / "MolecularNodesCache"
    bcif_path = cache_dir / "6Y0G.bcif"
    if not bcif_path.exists():
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


def main():
    global FOLD_DATA, SCROLL_PER_CYCLE, SCROLL_VECTOR

    mn.register()

    print(f"=== Loading scene ({N_CYCLES} cycles x {FRAMES_PER_CYCLE} = "
          f"{TOTAL_FRAMES} frames @ {FPS}fps) ===")

    # --- Load fold data for polypeptide morph ---
    fold_npz = "repeating_polypeptide_folds.npz"
    if os.path.exists(fold_npz):
        FOLD_DATA = dict(np.load(fold_npz, allow_pickle=True))
        repeat_distance = float(FOLD_DATA['repeat_distance'])
        SCROLL_PER_CYCLE = repeat_distance / N_CYCLES
        SCROLL_VECTOR = FOLD_DATA['scroll_vector']
        print(f"  Fold data: {fold_npz}")
        print(f"    {int(FOLD_DATA['n_domains'])} domains, "
              f"repeat_distance={repeat_distance:.1f}A, "
              f"scroll_per_cycle={SCROLL_PER_CYCLE:.2f}A")
    else:
        print(f"  WARNING: {fold_npz} not found, folding morph disabled")
        FOLD_DATA = None

    canvas = mn.Canvas(mn.scene.Cycles(samples=SAMPLES), resolution=RES)
    scene = bpy.context.scene
    scene.render.film_transparent = False
    set_bg(scene, (0.04, 0.04, 0.06), 0.5)
    scene.cycles.max_bounces = 12
    scene.cycles.transparent_max_bounces = 64
    scene.cycles.use_denoising = True

    # --- Load molecules ---
    print("  Loading molecules...")

    _extract_trna_pdbs()

    # Write backbone-only mRNA PDB for cleaner rendering
    _write_backbone("extended_mrna.pdb", "extended_mrna_bb.pdb", mol_type="rna")

    # Flat opaque material for ribosome silhouette pass (rendered with
    # film_transparent=True, then PIL edge-detects the alpha channel)
    def make_ribo_material():
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

    # 1. Ribosome (40S + 60S) — cartoon for silhouette pass
    mol_surface = mn.Molecule.fetch("6Y0G")
    mol_surface.add_style(
        style="cartoon",
        selection=mol_surface.select.chain_id(RIBOSOME_CHAINS),
        material=make_ribo_material(),
        name="surface",
    )

    # 2. mRNA (backbone-only for clean rendering)
    mol_mrna = mn.Molecule.load("extended_mrna_bb.pdb")
    mol_mrna.add_style(
        style="cartoon",
        material=make_solid_material((0.05, 0.25, 0.95)),
        name="mRNA",
    )

    # 3. P-site tRNA (chain B4) — ribbon
    mol_trna_p = mn.Molecule.load("trna_b4.pdb")
    mol_trna_p.add_style(
        style="ribbon",
        material=make_solid_material((0.95, 0.4, 0.0)),
        name="tRNA_P",
    )

    # 4. A-site tRNA (chain D4) — ribbon
    mol_trna_a = mn.Molecule.load("trna_d4.pdb")
    mol_trna_a.add_style(
        style="ribbon",
        material=make_solid_material((0.95, 0.4, 0.0)),
        name="tRNA_A",
    )

    # 5. Polypeptide (repeating domains) — spheres
    peptide_pdb = "repeating_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        peptide_pdb = "tunnel_polypeptide.pdb"
    if not os.path.exists(peptide_pdb):
        print(f"  WARNING: no polypeptide PDB found, falling back to extended_polypeptide.pdb")
        peptide_pdb = "extended_polypeptide.pdb"
    mol_peptide = mn.Molecule.load(peptide_pdb)
    mol_peptide.add_style(
        style="spheres",
        material=make_solid_material((0.85, 0.05, 0.55)),
        name="polypeptide",
    )

    # --- Find Blender objects ---
    def find_mesh(name_substr):
        return [o for o in bpy.data.objects if name_substr in o.name and o.type == "MESH"]

    objs_surface = find_mesh("6Y0G")
    objs_mrna = find_mesh("extended_mrna_bb")
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

    # Apply rotation to all primary objects
    primary_objects = [obj_surface, obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    for o in primary_objects:
        o.rotation_euler.z = math.pi / 2

    # --- Store original vertex positions for per-residue deformation ---
    deform_objects = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]
    orig_verts = {}
    for obj in deform_objects:
        n = len(obj.data.vertices)
        co = np.empty(n * 3, dtype=np.float64)
        obj.data.vertices.foreach_get('co', co)
        orig_verts[obj.name] = co.reshape(-1, 3).copy()
        print(f"  Stored {n} vertices for {obj.name}")

    # --- Apply mRNA bend (organic curvature outside ribosome) ---
    mrna_mesh_res_ids = get_mesh_res_ids(obj_mrna)
    print(f"  Applying mRNA bend (channel ±{MRNA_CHANNEL_HALF_LEN} BU, "
          f"strength {MRNA_BEND_STRENGTH})...")
    orig_verts[obj_mrna.name] = apply_mrna_bend(
        orig_verts[obj_mrna.name], mrna_mesh_res_ids)
    # Write bent positions back to mesh so first frame renders correctly
    obj_mrna.data.vertices.foreach_set('co', orig_verts[obj_mrna.name].ravel())
    obj_mrna.data.update()

    # --- Get mesh res_ids for MD mapping ---
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

    # --- Load ribosome coords for MD wall repulsion ---
    import time as _time
    t_md_start = _time.time()
    ribo_coords_A = None
    try:
        from biotite.structure import AtomArrayStack
        import biotite.structure.io.pdbx as pdbx
        from pathlib import Path
        cache_dir = Path.home() / "MolecularNodesCache"
        bcif_path = cache_dir / "6Y0G.bcif"
        if not bcif_path.exists():
            import biotite.database.rcsb as rcsb_db
            rcsb_db.fetch("6Y0G", "bcif", target_path=str(cache_dir))
        cif = pdbx.BinaryCIFFile.read(str(bcif_path))
        full_arr = pdbx.get_structure(cif, model=1)
        if isinstance(full_arr, AtomArrayStack):
            full_arr = full_arr[0]
        mask_ribo = np.isin(full_arr.chain_id, RIBOSOME_CHAINS)
        ribo_coords_A = full_arr[mask_ribo].coord
        print(f"  Ribosome context: {len(ribo_coords_A)} atoms for wall repulsion")
    except Exception as e:
        print(f"  WARNING: Could not load ribosome coords ({e}), MD without wall repulsion")

    # --- Initialize per-frame MD simulations for mobile molecules ---
    print("  Initializing OpenMM MD simulations...")
    md_sims = {}
    for name, pdb, mol_type in [
        ('mrna', 'extended_mrna.pdb', 'rna'),
        ('trna_p', 'trna_b4.pdb', 'rna'),
        ('trna_a', 'trna_d4.pdb', 'rna'),
        # Polypeptide PDB is backbone-only — can't parameterize for MD.
        # The folding morph animation provides its motion instead.
    ]:
        try:
            md_sims[name] = MolecularDynamics(
                name, pdb, ribo_coords_A, mol_type=mol_type)
        except Exception as e:
            print(f"  WARNING: {name} MD failed ({e}), static fallback")
            md_sims[name] = None

    # --- Load pre-computed ribosome thermal motion (if available) ---
    ribo_thermal = None
    ribo_thermal_res_ids = None
    ribo_thermal_path = "ribosome_thermal.npz"
    if os.path.exists(ribo_thermal_path):
        ribo_data = np.load(ribo_thermal_path)
        ribo_thermal = ribo_data['deltas']  # (n_frames, n_residues, 3) in BU
        ribo_thermal_res_ids = ribo_data['residue_ids']
        print(f"  Ribosome thermal: {ribo_thermal.shape[0]} frames, "
              f"{ribo_thermal.shape[1]} residues from {ribo_thermal_path}")
    else:
        print(f"  Ribosome thermal: {ribo_thermal_path} not found, ribosome static")

    # Store ribosome orig verts if thermal motion available
    if ribo_thermal is not None:
        n_ribo = len(obj_surface.data.vertices)
        co_ribo = np.empty(n_ribo * 3, dtype=np.float64)
        obj_surface.data.vertices.foreach_get('co', co_ribo)
        orig_verts[obj_surface.name] = co_ribo.reshape(-1, 3).copy()
        surface_res_ids = get_mesh_res_ids(obj_surface)
        print(f"  Stored {n_ribo} ribosome vertices for thermal deformation")

    dt_md = _time.time() - t_md_start
    print(f"  MD initialization: {dt_md:.1f}s")

    # --- Orthographic camera (matched to render_single_frame.py) ---
    import mathutils
    canvas.frame_object(mol_surface)
    cam = scene.camera
    # Rotation from Blender viewport (see render_single_frame.py docstring)
    cam_rot = mathutils.Euler((2.2480, 0.0, 0.0489), 'XYZ')
    target = mathutils.Vector((-2.66, 1.71, 1.72))
    forward = mathutils.Vector((0, 0, -1))
    forward.rotate(cam_rot)
    cam.location = target - forward * 50.0
    cam.rotation_euler = cam_rot
    cam.data.type = 'ORTHO'
    cam.data.ortho_scale = 10.5  # slightly wider than 9.0 for jitter margin
    cam.data.shift_x = 0.0
    cam.data.shift_y = 0.0
    print(f"  Camera (ortho): scale={cam.data.ortho_scale}, "
          f"rot={tuple(cam.rotation_euler)}")

    # --- Save .blend for GPU rendering (scene only, no animation) ---
    if SAVE_BLEND:
        blend_path = os.path.abspath("scene_animate.blend")
        print(f"  Saving animation scene to {blend_path}...")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        size_mb = os.path.getsize(blend_path) / (1024 * 1024)
        print(f"  Saved: {blend_path} ({size_mb:.1f} MB)")
        print("=== Done (scene saved, no render) ===")
        return

    # --- 2-pass composite constants ---
    OUTLINE_COLOR = (70, 120, 200)
    OUTLINE_THICKNESS = 3
    internal_objs = [obj_mrna, obj_trna_p, obj_trna_a, obj_peptide]

    # --- Render loop (2-pass composite per frame) ---
    import time
    print(f"\n=== Rendering {TOTAL_FRAMES} frames ({N_CYCLES} cycles) ===")

    for cycle in range(N_CYCLES):
        for local_frame in range(FRAMES_PER_CYCLE):
            global_frame = cycle * FRAMES_PER_CYCLE + local_frame

            # Skip already-rendered frames (resume support)
            frame_out = os.path.join(FRAMES_DIR, f"frame_{global_frame:04d}.png")
            if os.path.exists(frame_out):
                print(f"  Skipping frame {global_frame} (already exists)")
                continue

            t_frame_start = time.time()
            print(f"\n--- Cycle {cycle}, Frame {local_frame}/{FRAMES_PER_CYCLE - 1} "
                  f"(global {global_frame}/{TOTAL_FRAMES - 1}) ---")

            # Single-cycle choreographic deltas
            _, trna_p_d, trna_a_d = get_positions(local_frame)

            # --- Ribosome: static pose + optional pre-computed thermal ---
            obj_surface.location = (0, 0, 0)
            obj_surface.rotation_euler = (0, 0, math.pi / 2)
            if ribo_thermal is not None:
                ribo_frame = global_frame % len(ribo_thermal)
                ribo_deltas = ribo_thermal[ribo_frame]
                ribo_pos = apply_md_deltas_to_mesh(
                    orig_verts[obj_surface.name], surface_res_ids,
                    ribo_deltas, ribo_thermal_res_ids)
                obj_surface.data.vertices.foreach_set('co', ribo_pos.ravel())
                obj_surface.data.update()

            # --- mRNA: stationary + MD thermal ---
            obj_mrna.location = (0, 0, 0)
            obj_mrna.rotation_euler = (0, 0, math.pi / 2)

            # MD thermal deformation for mRNA
            md_mrna = md_sims.get('mrna')
            if md_mrna is not None:
                raw_deltas = md_mrna.step_and_get_deltas(global_frame)
                mrna_deltas = md_mrna.get_blended_deltas(
                    global_frame, TOTAL_FRAMES, raw_deltas)
                mrna_pos = apply_md_deltas_to_mesh(
                    orig_verts[obj_mrna.name], mrna_mesh_res_ids,
                    mrna_deltas, md_mrna.residue_ids)
            else:
                mrna_pos = orig_verts[obj_mrna.name]
            obj_mrna.data.vertices.foreach_set('co', mrna_pos.ravel())
            obj_mrna.data.update()

            # --- P-site tRNA: choreographic position + MD thermal ---
            obj_trna_p.location = tuple(trna_p_d)
            obj_trna_p.rotation_euler = (0, 0, math.pi / 2)

            md_trna_p = md_sims.get('trna_p')
            if md_trna_p is not None:
                raw_deltas = md_trna_p.step_and_get_deltas(global_frame)
                trna_p_deltas = md_trna_p.get_blended_deltas(
                    global_frame, TOTAL_FRAMES, raw_deltas)
                trna_p_pos = apply_md_deltas_to_mesh(
                    orig_verts[obj_trna_p.name], trna_p_mesh_res_ids,
                    trna_p_deltas, md_trna_p.residue_ids)
            else:
                trna_p_pos = orig_verts[obj_trna_p.name]
            obj_trna_p.data.vertices.foreach_set('co', trna_p_pos.ravel())
            obj_trna_p.data.update()

            # --- A-site tRNA: choreographic position + MD thermal ---
            obj_trna_a.location = tuple(trna_a_d)
            obj_trna_a.rotation_euler = (0, 0, math.pi / 2)

            md_trna_a = md_sims.get('trna_a')
            if md_trna_a is not None:
                raw_deltas = md_trna_a.step_and_get_deltas(global_frame)
                trna_a_deltas = md_trna_a.get_blended_deltas(
                    global_frame, TOTAL_FRAMES, raw_deltas)
                trna_a_pos = apply_md_deltas_to_mesh(
                    orig_verts[obj_trna_a.name], trna_a_mesh_res_ids,
                    trna_a_deltas, md_trna_a.residue_ids)
            else:
                trna_a_pos = orig_verts[obj_trna_a.name]
            obj_trna_a.data.vertices.foreach_set('co', trna_a_pos.ravel())
            obj_trna_a.data.update()

            # --- Polypeptide: morph + MD thermal ---
            obj_peptide.location = (0, 0, 0)
            obj_peptide.rotation_euler = (0, 0, math.pi / 2)

            pep_orig = orig_verts[obj_peptide.name]
            if FOLD_DATA is not None:
                morphed = compute_polypeptide_morph(
                    pep_orig, pep_res_ids, FOLD_DATA,
                    cycle, local_frame, FRAMES_PER_CYCLE)
            else:
                morphed = compute_peptide_positions(
                    pep_orig, pep_res_ids, cycle, local_frame)

            # Add MD thermal on top of morph
            md_pep = md_sims.get('peptide')
            if md_pep is not None:
                raw_deltas = md_pep.step_and_get_deltas(global_frame)
                pep_deltas = md_pep.get_blended_deltas(
                    global_frame, TOTAL_FRAMES, raw_deltas)
                morphed = apply_md_deltas_to_mesh(
                    morphed, pep_res_ids, pep_deltas, md_pep.residue_ids)
            obj_peptide.data.vertices.foreach_set('co', morphed.ravel())
            obj_peptide.data.update()

            bpy.context.view_layer.update()

            # --- 2-pass composite render ---
            frame_path = os.path.join(FRAMES_DIR, f"frame_{global_frame:04d}.png")
            pass1_path = os.path.join(FRAMES_DIR, "_pass_ribo.png")
            pass2_path = os.path.join(FRAMES_DIR, "_pass_internal.png")

            # Pass 1: Ribosome silhouette (transparent bg → alpha = shape mask)
            for o in internal_objs:
                o.hide_render = True
            obj_surface.hide_render = False
            scene.render.film_transparent = True
            canvas.snapshot(pass1_path)

            # Pass 2: Internal components (no ribosome, dark bg)
            obj_surface.hide_render = True
            for o in internal_objs:
                o.hide_render = False
            scene.render.film_transparent = False
            canvas.snapshot(pass2_path)

            # Restore visibility for next frame
            obj_surface.hide_render = False

            # Composite: edge-detect ribosome alpha → outline overlay
            internal_img = Image.open(pass2_path).convert("RGBA")
            ribo_img = Image.open(pass1_path).convert("RGBA")
            alpha = np.array(ribo_img)[:, :, 3]
            mask_arr = (alpha > 10).astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_arr).filter(
                ImageFilter.GaussianBlur(radius=2))
            mask_img = Image.fromarray(
                (np.array(mask_img) > 128).astype(np.uint8) * 255)
            edges = mask_img.filter(ImageFilter.FIND_EDGES)
            sil = Image.fromarray((np.array(edges) > 30).astype(np.uint8) * 255)
            for _ in range(OUTLINE_THICKNESS // 2):
                sil = sil.filter(ImageFilter.MaxFilter(3))
            edges_np = np.array(sil)
            overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
            edge_mask = edges_np > 100
            overlay[edge_mask, 0] = OUTLINE_COLOR[0]
            overlay[edge_mask, 1] = OUTLINE_COLOR[1]
            overlay[edge_mask, 2] = OUTLINE_COLOR[2]
            overlay[edge_mask, 3] = 255
            result = Image.alpha_composite(internal_img,
                                           Image.fromarray(overlay, "RGBA"))
            result.save(frame_path)

            dt = time.time() - t_frame_start
            print(f"  Frame saved: {frame_path} [{dt:.1f}s]")

    print(f"\n=== Done! {TOTAL_FRAMES} frames rendered to {FRAMES_DIR}/ ===")
    print("Next: python3.11 encode.py [--debug]")


if __name__ == "__main__":
    main()
