"""Build a polypeptide threaded through the ribosome exit tunnel with repeating domains.

Traces the exit tunnel void space from the peptidyl transferase center (PTC)
through the 60S subunit, then builds a polypeptide chain along the centerline.
Uses extended conformation inside the tunnel (~3.3 A/residue, fits the ~10 A
L4-L22 constriction). After the tunnel exit, places repeating Villin HP35
(1YRF) folded domains with 3-residue GSG linkers for visible folding animation.

Chain layout:
  [tunnel: ~30 res extended] [domain_0: 35 res] [GSG linker] [domain_1: 35 res]
  [GSG linker] [domain_2: 35 res] [GSG linker] [domain_3: 35 res] [tail: ~10 res]
  Total: ~192 residues

Outputs:
  - repeating_polypeptide.pdb — full chain in extended conformation
  - repeating_polypeptide_folds.npz — per-domain folded coordinates + metadata

Run with: python3.11 build_tunnel_polypeptide.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
from biotite.structure import AtomArrayStack, AtomArray, BondList, superimpose, concatenate
from biotite.structure.io.pdb import PDBFile
import biotite.database.rcsb as rcsb_db
import biotite.structure.io.pdbx as pdbx
import sys
import os
import tempfile

OUTPUT_PDB = "repeating_polypeptide.pdb"
OUTPUT_NPZ = "repeating_polypeptide_folds.npz"
# Also write legacy name for backward compat with render_single_frame.py
OUTPUT_LEGACY = "tunnel_polypeptide.pdb"

EXTENDED_RISE_PER_RESIDUE = 3.3  # Angstroms (extended conformation in tunnel)
HELIX_RISE_PER_RESIDUE = 1.5    # Angstroms (alpha helix after exit)
TRACE_STEP = 2.0  # Angstroms per tracing step
MIN_CLEARANCE = 4.0  # minimum distance from tunnel wall (A)
EXIT_THRESHOLD = 15.0  # distance at which we consider having exited
N_EXTENSION_STEPS = 400  # extension steps past tunnel exit (800 A = 80 BU)
EXTENSION_ANGULAR_STD = 0.15  # radians std dev for random-walk direction perturbation

# HP35 domain parameters
HP35_PDB = "1YRF"
DOMAIN_RESIDUES = 35
LINKER_RESIDUES = 3
REPEAT_UNIT = DOMAIN_RESIDUES + LINKER_RESIDUES  # 38 residues
N_DOMAINS = 8
TAIL_RESIDUES = 10

CA_CA_DIST = 3.8  # standard protein CA-CA distance (Angstroms)

# 60S chain IDs
CHAINS_60S = [
    "L5", "L7", "L8", "LA", "LB", "LC", "LD", "LE", "LF", "LG", "LH", "LI",
    "LJ", "LL", "LM", "LN", "LO", "LP", "LQ", "LR", "LS", "LT", "LU", "LV",
    "LW", "LX", "LY", "LZ", "La", "Lb", "Lc", "Ld", "Le", "Lg", "Lh",
    "Li", "Lj", "Lk", "Ll", "Lm", "Ln", "Lo", "Lp", "Lr",
]

# Ideal alpha helix parameters
PHI = np.radians(-57)
PSI = np.radians(-47)
OMEGA = np.radians(180)
N_CA_LEN = 1.458
CA_C_LEN = 1.523
C_N_LEN = 1.329
N_CA_C_ANGLE = np.radians(111.0)
CA_C_N_ANGLE = np.radians(116.6)
C_N_CA_ANGLE = np.radians(121.7)


def rotation_matrix(axis, theta):
    """Rodrigues rotation formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def place_atom(prev2, prev1, bond_len, bond_angle, dihedral):
    """Place atom given two predecessors, bond length, angle, and dihedral."""
    bc = prev1 - prev2
    bc_hat = bc / np.linalg.norm(bc)
    d = bc_hat * bond_len
    if abs(bc_hat[0]) < 0.9:
        perp = np.cross(bc_hat, np.array([1, 0, 0]))
    else:
        perp = np.cross(bc_hat, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)
    R_angle = rotation_matrix(perp, -(np.pi - bond_angle))
    d = R_angle @ d
    R_dihed = rotation_matrix(bc_hat, dihedral)
    d = R_dihed @ d
    return prev1 + d


def load_ribosome():
    """Load 6Y0G and return 60S atoms and C4 chain."""
    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    mol = mn.Molecule.fetch("6Y0G")
    arr = mol.array
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    # 60S subunit atoms
    mask_60s = np.isin(arr.chain_id, CHAINS_60S)
    atoms_60s = arr[mask_60s]
    print(f"  60S subunit: {len(atoms_60s)} atoms")

    # C4 chain (nascent peptide at PTC)
    c4 = arr[arr.chain_id == "C4"]
    print(f"  Chain C4: {len(c4)} atoms, {len(np.unique(c4.res_id))} residues")

    return atoms_60s, c4, arr


def find_ptc_position(c4):
    """Get the PTC position from C4 chain (last CA atom = growing end)."""
    ca_mask = c4.atom_name == "CA"
    ca_coords = c4.coord[ca_mask]
    ca_res_ids = c4.res_id[ca_mask]

    # The last residue's CA is at the PTC (growing end)
    last_idx = np.argmax(ca_res_ids)
    ptc = ca_coords[last_idx]
    print(f"  PTC position (last CA): ({ptc[0]:.1f}, {ptc[1]:.1f}, {ptc[2]:.1f})")
    return ptc


def trace_tunnel(atoms_60s_coords, ptc_pos, c4_ca_coords=None,
                 initial_direction=None):
    """Trace the exit tunnel through the 60S subunit.

    Uses the crystallographic C4 nascent chain backbone (if provided) as the
    ground-truth tunnel path. Falls back to greedy void-space tracing if C4
    doesn't reach exit. After exit, extends with a curving random walk.

    Returns: (centerline, exit_arc_length)
    """
    tree = KDTree(atoms_60s_coords)

    com = atoms_60s_coords.mean(axis=0)
    if initial_direction is None:
        initial_direction = ptc_pos - com
        initial_direction = initial_direction / np.linalg.norm(initial_direction)

    centerline = [ptc_pos.copy()]
    current_pos = ptc_pos.copy()
    current_dir = initial_direction.copy()
    exited = False

    # Phase 1: Follow crystallographic C4 backbone
    if c4_ca_coords is not None and len(c4_ca_coords) >= 2:
        print(f"  Phase 1: Following C4 crystallographic path "
              f"({len(c4_ca_coords)} CAs, PTC -> exit)...")
        for i in range(1, len(c4_ca_coords)):
            pt = c4_ca_coords[i]
            centerline.append(pt.copy())
            dist, _ = tree.query(pt)

            if i % 5 == 0 or i == len(c4_ca_coords) - 1:
                print(f"    C4 CA {i}/{len(c4_ca_coords)-1}: clearance={dist:.1f}A, "
                      f"pos=({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f})")

            if dist > EXIT_THRESHOLD:
                print(f"    C4 CA {i}: clearance {dist:.1f}A > {EXIT_THRESHOLD}A "
                      f"-- tunnel exit reached via C4 path")
                exited = True
                break

        current_pos = centerline[-1].copy()
        if len(centerline) >= 2:
            current_dir = centerline[-1] - centerline[-2]
            current_dir = current_dir / np.linalg.norm(current_dir)

        if not exited:
            print(f"  C4 path ended inside tunnel after {len(c4_ca_coords)} CAs, "
                  f"continuing with greedy tracer...")

    # Phase 2: Greedy void-space tracer
    FORWARD_BIAS = 3.0
    if not exited:
        max_steps = 200
        n_candidates = 36
        print(f"  Phase 2: Greedy void-space tracer (step={TRACE_STEP}A, "
              f"forward_bias={FORWARD_BIAS}, exit={EXIT_THRESHOLD}A)...")

        for step in range(max_steps):
            ahead = current_pos + current_dir * TRACE_STEP

            if abs(current_dir[0]) < 0.9:
                u = np.cross(current_dir, np.array([1, 0, 0]))
            else:
                u = np.cross(current_dir, np.array([0, 1, 0]))
            u = u / np.linalg.norm(u)
            v = np.cross(current_dir, u)

            best_pos = None
            best_score = -1
            best_clearance = -1

            for i in range(n_candidates):
                angle = 2 * np.pi * i / n_candidates
                for radius_frac in [0.0, 0.3, 0.6, 1.0]:
                    r = MIN_CLEARANCE * radius_frac
                    candidate = ahead + r * (np.cos(angle) * u + np.sin(angle) * v)
                    dist, _ = tree.query(candidate)
                    if dist < MIN_CLEARANCE * 0.5:
                        continue
                    disp = candidate - current_pos
                    disp_norm = np.linalg.norm(disp)
                    if disp_norm > 0:
                        alignment = np.dot(disp / disp_norm, current_dir)
                    else:
                        alignment = 0.0
                    score = dist + FORWARD_BIAS * alignment * dist
                    if score > best_score:
                        best_score = score
                        best_clearance = dist
                        best_pos = candidate

            if best_pos is None or best_clearance < MIN_CLEARANCE * 0.5:
                print(f"    Step {step}: no viable candidate, stopping")
                break

            if best_clearance > EXIT_THRESHOLD:
                print(f"    Step {step}: clearance {best_clearance:.1f}A "
                      f"> {EXIT_THRESHOLD}A, tunnel exit reached")
                centerline.append(best_pos)
                current_pos = best_pos
                new_dir = best_pos - centerline[-2]
                current_dir = new_dir / np.linalg.norm(new_dir)
                exited = True
                break

            centerline.append(best_pos)
            new_dir = best_pos - current_pos
            new_dir = new_dir / np.linalg.norm(new_dir)
            current_dir = 0.9 * current_dir + 0.1 * new_dir
            current_dir = current_dir / np.linalg.norm(current_dir)
            current_pos = best_pos

            if step % 20 == 0:
                print(f"    Step {step}: clearance={best_clearance:.1f}A, "
                      f"pos=({current_pos[0]:.1f}, {current_pos[1]:.1f}, "
                      f"{current_pos[2]:.1f})")

    exit_arc_length = None
    if exited:
        cl_arr = np.array(centerline)
        exit_arc_length = sum(np.linalg.norm(cl_arr[i+1] - cl_arr[i])
                              for i in range(len(cl_arr) - 1))

    # Phase 3: Random-walk extension past tunnel exit
    if exited:
        print(f"  Phase 3: Random-walk extension ({N_EXTENSION_STEPS} steps, "
              f"{N_EXTENSION_STEPS * TRACE_STEP:.0f}A)...")
        rng = np.random.default_rng(42)
        ext_pos = current_pos.copy()
        ext_dir = current_dir.copy()
        for ext in range(1, N_EXTENSION_STEPS + 1):
            if abs(ext_dir[0]) < 0.9:
                perp1 = np.cross(ext_dir, np.array([1, 0, 0]))
            else:
                perp1 = np.cross(ext_dir, np.array([0, 1, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(ext_dir, perp1)
            dtheta = rng.normal(0, EXTENSION_ANGULAR_STD)
            dphi = rng.uniform(0, 2 * np.pi)
            ext_dir = (ext_dir * np.cos(dtheta)
                       + perp1 * np.sin(dtheta) * np.cos(dphi)
                       + perp2 * np.sin(dtheta) * np.sin(dphi))
            ext_dir = ext_dir / np.linalg.norm(ext_dir)
            ext_pos = ext_pos + ext_dir * TRACE_STEP
            centerline.append(ext_pos.copy())

    centerline = np.array(centerline)
    total_length = sum(np.linalg.norm(centerline[i+1] - centerline[i])
                       for i in range(len(centerline) - 1))
    print(f"  Tunnel centerline: {len(centerline)} points, {total_length:.1f}A total length")
    if exit_arc_length is not None:
        print(f"  Tunnel exit at arc length: {exit_arc_length:.1f}A")
    return centerline, exit_arc_length


def smooth_centerline(centerline, exit_arc_length):
    """Smooth the centerline with cubic spline and variable-rate resampling.

    Inside tunnel: resample at EXTENDED_RISE_PER_RESIDUE (3.3 A).
    After exit: resample at HELIX_RISE_PER_RESIDUE (1.5 A).

    Returns: (spline_points, tunnel_exit_residue, spline_fn, total_len)
    """
    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(centerline))
    t[1:] = np.cumsum(seg_lengths)
    total_len = t[-1]

    cs = CubicSpline(t, centerline)

    if exit_arc_length is None:
        exit_arc_length = total_len

    tunnel_t = np.arange(0, exit_arc_length, EXTENDED_RISE_PER_RESIDUE)
    tunnel_exit_residue = len(tunnel_t)

    remaining = total_len - exit_arc_length
    if remaining > 0:
        post_exit_t = np.arange(exit_arc_length, total_len, HELIX_RISE_PER_RESIDUE)
        if len(post_exit_t) > 0 and len(tunnel_t) > 0:
            if abs(post_exit_t[0] - tunnel_t[-1]) < 0.1:
                post_exit_t = post_exit_t[1:]
        all_t = np.concatenate([tunnel_t, post_exit_t])
    else:
        all_t = tunnel_t

    smooth = cs(all_t)
    n_tunnel = tunnel_exit_residue
    n_helix = len(smooth) - n_tunnel

    print(f"  Smoothed centerline: {len(smooth)} points "
          f"({n_tunnel} extended @{EXTENDED_RISE_PER_RESIDUE}A + "
          f"{n_helix} helix @{HELIX_RISE_PER_RESIDUE}A)")
    print(f"  Tunnel exit at residue index: {tunnel_exit_residue}")

    return smooth, tunnel_exit_residue, cs, total_len


def build_backbone_along_spline(spline_points, tunnel_exit_residue):
    """Build polyalanine backbone atoms along the spline centerline.

    Returns: AtomArray
    """
    n_res = len(spline_points)
    print(f"  Building {n_res}-residue polyalanine along tunnel spline...")
    print(f"    Residues 1-{tunnel_exit_residue}: extended conformation (tunnel)")
    print(f"    Residues {tunnel_exit_residue+1}-{n_res}: alpha helix (post-exit)")

    tangents = np.zeros_like(spline_points)
    tangents[0] = spline_points[1] - spline_points[0]
    tangents[-1] = spline_points[-1] - spline_points[-2]
    for i in range(1, n_res - 1):
        tangents[i] = spline_points[i + 1] - spline_points[i - 1]
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    atoms_per_res = 5  # N, CA, C, O, CB
    total_atoms = n_res * atoms_per_res
    arr = AtomArray(total_atoms)

    offsets_extended = {
        "N":  np.array([-0.40, -0.30, -1.65]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C":  np.array([0.40, 0.30, 1.65]),
        "O":  np.array([0.20, 1.20, 2.00]),
        "CB": np.array([-1.52, 0.0, 0.22]),
    }

    offsets_helix = {
        "N":  np.array([-0.53, -0.84, -0.75]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C":  np.array([0.53, 0.84, 0.75]),
        "O":  np.array([0.25, 1.96, 0.92]),
        "CB": np.array([-1.52, 0.0, 0.22]),
    }

    atom_names = ["N", "CA", "C", "O", "CB"]
    elements = ["N", "C", "C", "O", "C"]

    idx = 0
    for i in range(n_res):
        ca_pos = spline_points[i]
        t = tangents[i]

        if abs(t[0]) < 0.9:
            n_vec = np.cross(t, np.array([1, 0, 0]))
        else:
            n_vec = np.cross(t, np.array([0, 1, 0]))
        n_vec = n_vec / np.linalg.norm(n_vec)
        b_vec = np.cross(t, n_vec)

        if i < tunnel_exit_residue:
            offsets = offsets_extended
            twist_angle = np.radians(180) * i
        else:
            offsets = offsets_helix
            twist_angle = np.radians(100) * i

        cos_tw, sin_tw = np.cos(twist_angle), np.sin(twist_angle)
        n_rot = cos_tw * n_vec + sin_tw * b_vec
        b_rot = -sin_tw * n_vec + cos_tw * b_vec

        R = np.column_stack([n_rot, b_rot, t])

        for j, (aname, elem) in enumerate(zip(atom_names, elements)):
            pos = ca_pos + R @ offsets[aname]
            arr.coord[idx] = pos
            arr.atom_name[idx] = aname
            arr.res_name[idx] = "ALA"
            arr.res_id[idx] = i + 1
            arr.chain_id[idx] = "A"
            arr.element[idx] = elem
            arr.hetero[idx] = False
            idx += 1

    return arr


# ---------------------------------------------------------------------------
# HP35 domain handling
# ---------------------------------------------------------------------------
def fetch_hp35_domain():
    """Fetch Villin HP35 (1YRF) with full sidechains.

    Returns: AtomArray with all non-water protein atoms (backbone + sidechains).
    """
    print(f"  Fetching {HP35_PDB} (Villin HP35) for folded domain...")
    cif_path = rcsb_db.fetch(HP35_PDB, "cif", target_path="/tmp")
    cif = pdbx.CIFFile.read(cif_path)
    full_arr = pdbx.get_structure(cif, model=1)
    if isinstance(full_arr, AtomArrayStack):
        full_arr = full_arr[0]

    chains = np.unique(full_arr.chain_id)
    domain = full_arr[full_arr.chain_id == chains[0]]
    domain = domain[~domain.hetero]  # remove water and heteroatoms

    # Keep backbone + CB for consistent rendering with MolecularNodes
    backbone_names = {"N", "CA", "C", "O", "CB"}
    domain = domain[np.isin(domain.atom_name, list(backbone_names))]

    n_res = len(np.unique(domain.res_id))
    print(f"  HP35 domain: {HP35_PDB} chain {chains[0]}, "
          f"{n_res} residues, {len(domain)} backbone+CB atoms")
    return domain


def align_domain_to_position(domain, attach_pos, attach_dir):
    """Align a folded domain so its N-terminal CA is at attach_pos,
    oriented along attach_dir.

    Returns: modified domain AtomArray (copy).
    """
    domain = domain.copy()
    ca_mask = domain.atom_name == "CA"
    ca_coords = domain.coord[ca_mask]
    ca_res = domain.res_id[ca_mask]

    if len(ca_coords) < 2:
        return domain

    sort_idx = np.argsort(ca_res)
    n_term_ca = ca_coords[sort_idx[0]]
    c_term_ca = ca_coords[sort_idx[-1]]
    domain_axis = c_term_ca - n_term_ca
    domain_axis = domain_axis / np.linalg.norm(domain_axis)

    domain.coord -= n_term_ca

    attach_dir = attach_dir / np.linalg.norm(attach_dir)
    cross = np.cross(domain_axis, attach_dir)
    cross_norm = np.linalg.norm(cross)
    dot = np.dot(domain_axis, attach_dir)

    if cross_norm > 1e-6:
        R = rotation_matrix(cross / cross_norm, np.arccos(np.clip(dot, -1, 1)))
        domain.coord = (R @ domain.coord.T).T
    elif dot < 0:
        if abs(attach_dir[0]) < 0.9:
            perp = np.cross(attach_dir, np.array([1, 0, 0]))
        else:
            perp = np.cross(attach_dir, np.array([0, 1, 0]))
        perp = perp / np.linalg.norm(perp)
        R = rotation_matrix(perp, np.pi)
        domain.coord = (R @ domain.coord.T).T

    domain.coord += attach_pos
    return domain


def build_extended_segment(n_residues, start_pos, direction, res_name="ALA"):
    """Build an extended backbone segment along a direction.

    Returns: AtomArray with backbone atoms (N, CA, C, O, CB) per residue.
    """
    direction = direction / np.linalg.norm(direction)

    # Generate CA positions along the direction
    ca_positions = np.array([start_pos + i * CA_CA_DIST * direction
                              for i in range(n_residues)])

    # Use build_backbone_along_spline with all-extended conformation
    arr = build_backbone_along_spline(ca_positions, n_residues)

    # Overwrite residue names
    arr.res_name[:] = res_name
    return arr


def build_random_walk_extension(start_pos, start_dir, n_residues=10,
                                 ca_spacing=3.8, angular_std=0.10, seed=42,
                                 repel_center=None, repel_strength=0.3):
    """Build a smooth random-walk polyalanine extension.

    Returns: AtomArray
    """
    rng = np.random.default_rng(seed)
    ca_positions = [start_pos.copy()]
    current_dir = start_dir / np.linalg.norm(start_dir)

    for _ in range(n_residues):
        if abs(current_dir[0]) < 0.9:
            perp1 = np.cross(current_dir, np.array([1.0, 0.0, 0.0]))
        else:
            perp1 = np.cross(current_dir, np.array([0.0, 1.0, 0.0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(current_dir, perp1)

        dtheta = rng.normal(0, angular_std)
        dphi = rng.uniform(0, 2 * np.pi)
        current_dir = (current_dir * np.cos(dtheta)
                       + perp1 * np.sin(dtheta) * np.cos(dphi)
                       + perp2 * np.sin(dtheta) * np.sin(dphi))
        current_dir = current_dir / np.linalg.norm(current_dir)

        if repel_center is not None:
            away = ca_positions[-1] - repel_center
            away = away / np.linalg.norm(away)
            current_dir = (1 - repel_strength) * current_dir + repel_strength * away
            current_dir = current_dir / np.linalg.norm(current_dir)

        next_ca = ca_positions[-1] + current_dir * ca_spacing
        ca_positions.append(next_ca)

    ca_positions = np.array(ca_positions)
    return build_backbone_along_spline(ca_positions, 0)


def build_repeating_domain_chain(spline_points, tunnel_exit_residue,
                                  exit_pos, exit_dir, ribo_center=None):
    """Build polypeptide with repeating HP35 domains after the tunnel.

    Layout:
      1. Extended conformation in tunnel (polyalanine)
      2. N_DOMAINS x (HP35 folded domain + GSG linker)
      3. Short tail extension

    Returns: (combined AtomArray, fold_data dict)
    """
    print(f"\n=== Building repeating domain chain ===")
    print(f"  {N_DOMAINS} HP35 domains x {DOMAIN_RESIDUES} res + "
          f"{LINKER_RESIDUES} res linkers = {REPEAT_UNIT} res/repeat")

    # Build tunnel portion (extended polyalanine up to exit)
    tunnel_poly = build_backbone_along_spline(
        spline_points[:tunnel_exit_residue], tunnel_exit_residue)

    tunnel_ca = tunnel_poly.coord[tunnel_poly.atom_name == "CA"]
    tunnel_last_ca = tunnel_ca[-1]
    if len(tunnel_ca) >= 2:
        attach_dir = tunnel_ca[-1] - tunnel_ca[-2]
        attach_dir = attach_dir / np.linalg.norm(attach_dir)
    else:
        attach_dir = exit_dir

    n_tunnel_res = len(np.unique(tunnel_poly.res_id))
    print(f"  Tunnel: {n_tunnel_res} residues (extended)")

    # Bias exit direction away from ribosome and upward
    chain_dir = attach_dir.copy()
    if ribo_center is not None:
        away = tunnel_last_ca - ribo_center
        away = away / np.linalg.norm(away)
        upper_left = np.array([-0.5, 0.0, 1.0])
        upper_left = upper_left / np.linalg.norm(upper_left)
        chain_dir = 0.4 * chain_dir + 0.1 * away + 0.5 * upper_left
        chain_dir = chain_dir / np.linalg.norm(chain_dir)

    # Fetch HP35 template
    hp35_template = fetch_hp35_domain()
    hp35_template_ca = hp35_template.coord[hp35_template.atom_name == "CA"]
    hp35_n_res = len(np.unique(hp35_template.res_id))

    # Compute HP35 extent along its N->C axis for domain spacing
    hp35_res = hp35_template.res_id[hp35_template.atom_name == "CA"]
    sort_idx = np.argsort(hp35_res)
    hp35_n_term = hp35_template_ca[sort_idx[0]]
    hp35_c_term = hp35_template_ca[sort_idx[-1]]
    domain_extent = np.linalg.norm(hp35_c_term - hp35_n_term)
    print(f"  HP35 extent (N->C CA): {domain_extent:.1f}A")

    # Spacing between consecutive domain starts (domain extent + linker length)
    repeat_distance = domain_extent + LINKER_RESIDUES * CA_CA_DIST
    print(f"  Repeat distance: {repeat_distance:.1f}A "
          f"(domain {domain_extent:.1f}A + linker {LINKER_RESIDUES * CA_CA_DIST:.1f}A)")

    # Place domains along chain direction
    parts = [tunnel_poly]
    current_res_id = n_tunnel_res
    current_pos = tunnel_last_ca + CA_CA_DIST * chain_dir

    fold_data = {
        'n_domains': N_DOMAINS,
        'repeat_residues': REPEAT_UNIT,
        'scroll_vector': chain_dir.copy(),
        'repeat_distance': repeat_distance,
    }

    for di in range(N_DOMAINS):
        print(f"\n  Domain {di}:")

        # Place folded domain
        domain = align_domain_to_position(hp35_template, current_pos, chain_dir)
        domain_ca = domain.coord[domain.atom_name == "CA"]
        domain_res_orig = np.unique(domain.res_id)

        # Renumber domain residues
        current_res_id += 1
        domain_start_res = current_res_id
        res_remap = {int(old): domain_start_res + i
                     for i, old in enumerate(domain_res_orig)}
        for ai in range(len(domain)):
            domain.res_id[ai] = res_remap[int(domain.res_id[ai])]
        domain.chain_id[:] = "A"
        domain_end_res = current_res_id + len(domain_res_orig) - 1
        current_res_id = domain_end_res

        # Store folded coordinates for this domain
        fold_data[f'domain_{di}_folded'] = domain.coord.copy()
        fold_data[f'domain_{di}_start_res'] = domain_start_res
        fold_data[f'domain_{di}_end_res'] = domain_end_res
        fold_data[f'domain_{di}_atom_names'] = domain.atom_name.copy()

        # Build extended conformation for the same residues
        # (backbone stretched along chain direction)
        n_domain_res = len(domain_res_orig)
        ext_segment = build_extended_segment(
            n_domain_res, current_pos, chain_dir)
        # Match residue IDs to the folded domain
        ext_res_orig = np.unique(ext_segment.res_id)
        for ai in range(len(ext_segment)):
            old_res = ext_segment.res_id[ai]
            local_idx = np.searchsorted(ext_res_orig, old_res)
            ext_segment.res_id[ai] = domain_start_res + local_idx
        ext_segment.chain_id[:] = "A"

        fold_data[f'domain_{di}_extended'] = ext_segment.coord.copy()

        print(f"    Folded: res {domain_start_res}-{domain_end_res} "
              f"({n_domain_res} res, {len(domain)} atoms)")
        print(f"    Attach: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, "
              f"{current_pos[2]:.1f})")

        # Use FOLDED domain for PDB output (visible secondary structure)
        # Animation morph will interpolate between extended/folded from NPZ
        parts.append(domain)

        # Advance position based on folded domain's C-terminal CA
        domain_ca_sorted = domain.coord[domain.atom_name == "CA"]
        domain_ca_res = domain.res_id[domain.atom_name == "CA"]
        c_term_idx = np.argmax(domain_ca_res)
        current_pos = domain_ca_sorted[c_term_idx] + CA_CA_DIST * chain_dir

        # Add GSG linker (except after last domain)
        if di < N_DOMAINS - 1:
            linker = build_extended_segment(
                LINKER_RESIDUES, current_pos, chain_dir, res_name="GLY")
            current_res_id += 1
            linker_start = current_res_id
            linker_res_orig = np.unique(linker.res_id)
            for ai in range(len(linker)):
                old_res = linker.res_id[ai]
                local_idx = np.searchsorted(linker_res_orig, old_res)
                linker.res_id[ai] = linker_start + local_idx
            linker.chain_id[:] = "A"
            # Set GSG residue names
            linker_res_unique = np.unique(linker.res_id)
            for li, lr in enumerate(linker_res_unique):
                name = ["GLY", "SER", "GLY"][li % 3]
                linker.res_name[linker.res_id == lr] = name
            current_res_id = linker_start + LINKER_RESIDUES - 1
            parts.append(linker)

            linker_ca = linker.coord[linker.atom_name == "CA"]
            current_pos = linker_ca[-1] + CA_CA_DIST * chain_dir
            print(f"    Linker: {LINKER_RESIDUES} res (GSG)")

    # Tail extension (random walk off-screen)
    tail = build_random_walk_extension(
        current_pos, chain_dir, n_residues=TAIL_RESIDUES - 1,
        ca_spacing=CA_CA_DIST, angular_std=0.08, seed=99,
        repel_center=ribo_center, repel_strength=0.2)
    current_res_id += 1
    tail_start = current_res_id
    tail_res_orig = np.unique(tail.res_id)
    for ai in range(len(tail)):
        old_res = tail.res_id[ai]
        local_idx = np.searchsorted(tail_res_orig, old_res)
        tail.res_id[ai] = tail_start + local_idx
    tail.chain_id[:] = "A"
    parts.append(tail)
    print(f"\n  Tail: {TAIL_RESIDUES} residues (random walk)")

    # Concatenate all parts
    combined = concatenate(parts)
    n_total_res = len(np.unique(combined.res_id))
    print(f"\n  Total: {n_total_res} residues, {len(combined)} atoms")

    # Verify CA-CA junctions
    all_ca = combined.coord[combined.atom_name == "CA"]
    if len(all_ca) > 1:
        ca_dists = np.linalg.norm(np.diff(all_ca, axis=0), axis=1)
        bad = np.where((ca_dists > 6.0) | (ca_dists < 1.5))[0]
        if len(bad) > 0:
            print(f"  WARNING: {len(bad)} problematic CA-CA distances (>6A or <1.5A)")
            for i in bad[:10]:
                print(f"    CA {i+1}->{i+2}: {ca_dists[i]:.2f}A")
        else:
            print(f"  All CA-CA distances in range [1.5, 6.0]A")

    return combined, fold_data


def declash_structure(coords, ribosome_tree, ribo_coords, min_dist=3.0, max_iter=100):
    """Push polypeptide atoms away from ribosome walls."""
    print(f"  Geometric de-clash (min_dist={min_dist}A, max_iter={max_iter})...")
    for iteration in range(max_iter):
        dists, idxs = ribosome_tree.query(coords)
        clashing = dists < min_dist
        n_clash = clashing.sum()
        if n_clash == 0:
            print(f"    Converged at iteration {iteration}: no clashes")
            break
        push_dir = coords[clashing] - ribo_coords[idxs[clashing]]
        norms = np.linalg.norm(push_dir, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-6)
        push_dir = push_dir / norms
        deficit = min_dist - dists[clashing]
        coords[clashing] += push_dir * deficit[:, None]
        if iteration % 20 == 0:
            print(f"    Iteration {iteration}: {n_clash} clashing atoms, "
                  f"min dist={dists.min():.2f}A")
    return coords


def verify_clearance(label, coords, ribosome_tree):
    """Verify and print clearance stats."""
    distances, _ = ribosome_tree.query(coords)
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    n_below_25 = (distances < 2.5).sum()
    n_below_30 = (distances < 3.0).sum()
    pct_below_30 = 100.0 * n_below_30 / len(distances)

    print(f"\n=== Clearance: {label} ({len(coords)} atoms) ===")
    print(f"  Min: {min_dist:.2f}A  Max: {max_dist:.1f}A  Mean: {mean_dist:.1f}A")
    print(f"  Atoms < 2.5A: {n_below_25}  Atoms < 3.0A: {n_below_30} ({pct_below_30:.1f}%)")
    if n_below_25 == 0 and pct_below_30 < 5.0:
        print(f"  PASS: zero atoms < 2.5A, {pct_below_30:.1f}% < 3.0A")
    else:
        print(f"  WARN: acceptance criteria not met")
    return distances


def relax_polypeptide(polypeptide_pdb, ribo_coords, output_pdb):
    """3-stage constrained MD with wall repulsion."""
    from openmm.app import (
        PDBFile as OmmPDB, ForceField, Modeller, Simulation,
        NoCutoff, HBonds, StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator, CustomExternalForce
    from openmm.unit import kelvin, picosecond, picoseconds, kilojoule_per_mole, nanometer

    PHASE1_STEPS = 50000
    PHASE1_TEMP = 400
    PHASE2_STEPS = 30000
    PHASE2_TEMP = 350
    PHASE3_STEPS = 20000
    PHASE3_TEMP = 310
    TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS

    print(f"\n=== 3-stage MD relaxation with wall repulsion ({TOTAL_STEPS} steps) ===")

    with open(polypeptide_pdb) as f:
        lines = [line for line in f if not line.startswith("CONECT")]
    clean = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    clean.writelines(lines)
    clean.close()

    try:
        pdb = OmmPDB(clean.name)
    finally:
        os.unlink(clean.name)

    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")
    modeller = Modeller(pdb.topology, pdb.positions)

    try:
        modeller.addHydrogens(ff)
    except ValueError:
        print("  Adding hydrogens with explicit terminal variants...")
        residues = list(modeller.topology.residues())
        variants = [None] * len(residues)
        variants[0] = 'ACE'
        variants[-1] = 'NME'
        try:
            modeller.addHydrogens(ff, variants=variants)
        except (ValueError, KeyError):
            print("  WARNING: Could not add terminal caps, skipping MD relaxation")
            import shutil
            shutil.copy(polypeptide_pdb, output_pdb)
            print(f"  Copied raw PDB to {output_pdb}")
            return

    n_peptide_atoms = modeller.topology.getNumAtoms()
    print(f"  Polypeptide: {n_peptide_atoms} atoms (with H)")

    system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                             constraints=HBonds)

    # Position restraints
    restraint = CustomExternalForce("0.5*k_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k_restraint", 50.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.positions
    for i in range(n_peptide_atoms):
        pos = positions[i].value_in_unit(nanometer)
        restraint.addParticle(i, [pos[0], pos[1], pos[2]])
    system.addForce(restraint)

    # Wall repulsion force
    ribo_tree = KDTree(ribo_coords)
    pep_coords_nm = np.array([positions[i].value_in_unit(nanometer)
                               for i in range(n_peptide_atoms)])
    pep_coords_A = pep_coords_nm * 10.0

    _, nearest_idx = ribo_tree.query(pep_coords_A)
    nearest_ribo_A = ribo_coords[nearest_idx]
    nearest_ribo_nm = nearest_ribo_A * 0.1

    wall_force = CustomExternalForce(
        "0.5*k_wall*step(r_min-dist)*((r_min-dist)^2);"
        "dist=sqrt((x-wx)^2+(y-wy)^2+(z-wz)^2);"
        "r_min=0.3"
    )
    wall_force.addGlobalParameter("k_wall", 1000.0)
    wall_force.addPerParticleParameter("wx")
    wall_force.addPerParticleParameter("wy")
    wall_force.addPerParticleParameter("wz")

    for i in range(n_peptide_atoms):
        wall_force.addParticle(i, [
            nearest_ribo_nm[i, 0], nearest_ribo_nm[i, 1], nearest_ribo_nm[i, 2]
        ])
    system.addForce(wall_force)

    integrator = LangevinMiddleIntegrator(
        PHASE1_TEMP * kelvin, 1 / picosecond, 0.002 * picoseconds)
    from openmm import Platform
    platform = Platform.getPlatformByName('CPU')
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=0)

    print(f"  Phase 1: {PHASE1_STEPS} steps at {PHASE1_TEMP}K "
          f"(k_restraint=50, k_wall=1000)...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(PHASE1_STEPS // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(PHASE1_STEPS)

    print(f"  Phase 2: {PHASE2_STEPS} steps at {PHASE2_TEMP}K "
          f"(k_restraint=20, k_wall=3000)...")
    integrator.setTemperature(PHASE2_TEMP * kelvin)
    sim.context.setParameter("k_restraint", 20.0)
    sim.context.setParameter("k_wall", 3000.0)
    sim.step(PHASE2_STEPS)

    print(f"  Phase 3: {PHASE3_STEPS} steps at {PHASE3_TEMP}K "
          f"(k_restraint=10, k_wall=5000)...")
    integrator.setTemperature(PHASE3_TEMP * kelvin)
    sim.context.setParameter("k_restraint", 10.0)
    sim.context.setParameter("k_wall", 5000.0)
    sim.step(PHASE3_STEPS)

    print("  Final minimization...")
    sim.minimizeEnergy(maxIterations=0)
    state_final = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"  Final energy: {state_final.getPotentialEnergy()}")

    with open(output_pdb, "w") as f:
        OmmPDB.writeFile(sim.topology, state_final.getPositions(), f, keepIds=True)
    print(f"  Written: {output_pdb}")


def verify_channel_threading(coords, ribo_coords, ribo_tree, label="polypeptide"):
    """Verify that the molecule threads through the ribosome channel."""
    print(f"\n=== Channel threading verification: {label} ===")

    ribo_min = ribo_coords.min(axis=0)
    ribo_max = ribo_coords.max(axis=0)
    ribo_extent = ribo_max - ribo_min
    print(f"  Ribosome extent: ({ribo_extent[0]:.0f}, {ribo_extent[1]:.0f}, "
          f"{ribo_extent[2]:.0f}) A")

    mol_min = coords.min(axis=0)
    mol_max = coords.max(axis=0)
    mol_extent = mol_max - mol_min
    max_mol_extent = np.max(mol_extent)
    max_ribo_extent = np.max(ribo_extent)
    span_ratio = max_mol_extent / max_ribo_extent
    span_pass = span_ratio > 0.5
    print(f"  Molecule max extent: {max_mol_extent:.0f} A "
          f"({span_ratio:.1f}x ribosome)")
    print(f"  Span check: {'PASS' if span_pass else 'FAIL'} "
          f"(need > 0.5x ribosome)")

    distances, _ = ribo_tree.query(coords)
    inside_mask = distances < 15.0
    n_inside = inside_mask.sum()

    if n_inside > 0:
        inside_dists = distances[inside_mask]
        n_clipping = (inside_dists < 2.5).sum()
        clearance_pass = n_clipping == 0
        print(f"  Atoms inside ribosome (< 15 A from walls): {n_inside}")
        print(f"  Atoms clipping walls (< 2.5 A): {n_clipping}")
        print(f"  Clearance check: {'PASS' if clearance_pass else 'FAIL'}")
    else:
        clearance_pass = True
        print(f"  No atoms inside ribosome -- molecule is entirely outside")
        print(f"  Clearance check: PASS (trivially)")

    overall = span_pass and clearance_pass
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")
    return overall


def main():
    print("=== Building tunnel-threaded polypeptide with repeating HP35 domains ===")

    # Load ribosome
    atoms_60s, c4, full_arr = load_ribosome()
    ribo_coords = atoms_60s.coord
    ribo_tree = KDTree(ribo_coords)

    # Find PTC position
    ptc_pos = find_ptc_position(c4)

    # Get C4 CA coordinates sorted PTC -> exit (descending res_id)
    ca_mask = c4.atom_name == "CA"
    ca_coords = c4.coord[ca_mask]
    ca_res = c4.res_id[ca_mask]
    if len(ca_coords) >= 2:
        sort_idx_desc = np.argsort(ca_res)[::-1]
        c4_cas = ca_coords[sort_idx_desc]
        initial_dir = c4_cas[-1] - c4_cas[0]
        initial_dir = initial_dir / np.linalg.norm(initial_dir)
        print(f"  C4 chain: {len(c4_cas)} CAs, PTC -> exit")
        print(f"  Initial direction from C4: ({initial_dir[0]:.2f}, "
              f"{initial_dir[1]:.2f}, {initial_dir[2]:.2f})")
    else:
        c4_cas = None
        initial_dir = None

    # Trace tunnel
    centerline, exit_arc_length = trace_tunnel(
        ribo_coords, ptc_pos, c4_ca_coords=c4_cas, initial_direction=initial_dir)

    # Smooth centerline
    spline_points, tunnel_exit_residue, spline_fn, total_len = smooth_centerline(
        centerline, exit_arc_length)

    # Compute exit position and direction
    exit_pos = spline_points[tunnel_exit_residue]
    if tunnel_exit_residue + 1 < len(spline_points):
        exit_dir = spline_points[tunnel_exit_residue + 1] - spline_points[tunnel_exit_residue]
    elif tunnel_exit_residue > 0:
        exit_dir = spline_points[tunnel_exit_residue] - spline_points[tunnel_exit_residue - 1]
    else:
        exit_dir = np.array([0, 0, -1])
    exit_dir = exit_dir / np.linalg.norm(exit_dir)

    # Build repeating domain polypeptide
    ribo_center = ribo_coords.mean(axis=0)
    polypeptide, fold_data = build_repeating_domain_chain(
        spline_points, tunnel_exit_residue, exit_pos, exit_dir,
        ribo_center=ribo_center)

    # Geometric de-clash against ribosome walls
    verify_clearance("before de-clash", polypeptide.coord, ribo_tree)
    polypeptide.coord = declash_structure(
        polypeptide.coord, ribo_tree, ribo_coords, min_dist=3.0)
    verify_clearance("after de-clash", polypeptide.coord, ribo_tree)

    # Write raw PDB
    raw_pdb = "repeating_polypeptide_raw.pdb"
    pdb = PDBFile()
    pdb.set_structure(polypeptide)
    pdb.write(raw_pdb)
    print(f"  Raw polypeptide: {raw_pdb} ({len(polypeptide)} atoms, "
          f"{len(np.unique(polypeptide.res_id))} residues)")

    # Relax with 3-stage MD + wall repulsion
    try:
        relax_polypeptide(raw_pdb, ribo_coords, OUTPUT_PDB)
    except Exception as e:
        print(f"  WARNING: MD relaxation failed ({e}), using raw structure")
        import shutil
        shutil.copy(raw_pdb, OUTPUT_PDB)

    # Also write legacy name
    import shutil
    shutil.copy(OUTPUT_PDB, OUTPUT_LEGACY)
    print(f"  Also copied to {OUTPUT_LEGACY} (backward compat)")

    # Save fold data NPZ
    np.savez(OUTPUT_NPZ, **fold_data)
    print(f"  Fold data: {OUTPUT_NPZ}")
    for key in sorted(fold_data.keys()):
        val = fold_data[key]
        if isinstance(val, np.ndarray):
            print(f"    {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"    {key}: {val}")

    # Reload and verify
    final_pdb = PDBFile.read(OUTPUT_PDB)
    final_arr = final_pdb.get_structure()
    if isinstance(final_arr, AtomArrayStack):
        final_arr = final_arr[0]

    verify_clearance("final structure", final_arr.coord, ribo_tree)

    ca_coords = final_arr.coord[final_arr.atom_name == "CA"]
    if len(ca_coords) > 0:
        extent = np.linalg.norm(ca_coords[-1] - ca_coords[0])
        print(f"\n  Final polypeptide: {len(final_arr)} atoms, "
              f"{len(np.unique(final_arr.res_id))} residues")
        print(f"  CA extent: {extent:.1f}A = {extent * 0.1:.1f} BU")

    verify_channel_threading(final_arr.coord, ribo_coords, ribo_tree, "polypeptide")

    # Clean up temp file
    if os.path.exists(raw_pdb):
        os.unlink(raw_pdb)

    print("=== Done ===")


if __name__ == "__main__":
    main()
