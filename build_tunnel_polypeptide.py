"""Build a polypeptide threaded through the ribosome exit tunnel.

Traces the exit tunnel void space from the peptidyl transferase center (PTC)
through the 60S subunit, then builds a polyalanine chain along the centerline.
Uses extended conformation inside the tunnel (~3.3 A/residue, fits the ~10 A
L4-L22 constriction) and alpha helix after the exit (~1.5 A/residue).

After backbone construction, a geometric de-clash pushes atoms away from
ribosome walls, then 3-stage MD annealing with wall-repulsion forces relaxes
the structure while preventing clipping.

Algorithm:
  1. Build KDTree of all 60S ribosome atoms
  2. From C4 position (PTC), trace through void space by picking points
     with maximum clearance from ribosome walls
  3. Smooth centerline with variable-rate resampling (extended in tunnel,
     helix after exit)
  4. Build polyalanine backbone along spline with geometry switching
  5. Geometric de-clash against ribosome walls
  6. 3-stage constrained MD with wall repulsion (100K steps total)

Output: tunnel_polypeptide.pdb

Run with: python3.11 build_tunnel_polypeptide.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import CubicSpline
from biotite.structure import AtomArrayStack, AtomArray, BondList, superimpose
from biotite.structure.io.pdb import PDBFile
import sys
import os
import tempfile

OUTPUT = "tunnel_polypeptide.pdb"
EXTENDED_RISE_PER_RESIDUE = 3.3  # Angstroms (extended conformation in tunnel)
HELIX_RISE_PER_RESIDUE = 1.5    # Angstroms (alpha helix after exit)
TRACE_STEP = 2.0  # Angstroms per tracing step
MIN_CLEARANCE = 4.0  # minimum distance from tunnel wall (A)
EXIT_THRESHOLD = 15.0  # distance at which we consider having exited
N_EXTENSION_STEPS = 150  # extension steps past tunnel exit (300 A)

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


def trace_tunnel(atoms_60s_coords, ptc_pos, initial_direction=None):
    """Trace the exit tunnel through the 60S subunit.

    Starting from PTC, find the path through the tunnel by following
    maximum-clearance points at each step.

    Returns: (centerline, exit_arc_length)
        centerline: (N, 3) array of tunnel centerline points
        exit_arc_length: arc length at which tunnel exit was detected
    """
    tree = KDTree(atoms_60s_coords)

    # Estimate initial direction: away from ribosome center of mass
    com = atoms_60s_coords.mean(axis=0)
    if initial_direction is None:
        initial_direction = ptc_pos - com
        initial_direction = initial_direction / np.linalg.norm(initial_direction)

    centerline = [ptc_pos.copy()]
    current_pos = ptc_pos.copy()
    current_dir = initial_direction.copy()

    max_steps = 200
    n_candidates = 36  # sample points on disc
    exit_arc_length = None

    print(f"  Tracing tunnel (step={TRACE_STEP}A, "
          f"min_clearance={MIN_CLEARANCE}A, exit={EXIT_THRESHOLD}A)...")

    for step in range(max_steps):
        # Sample candidate points on a disc perpendicular to current direction
        # at distance TRACE_STEP ahead
        ahead = current_pos + current_dir * TRACE_STEP

        # Build orthonormal basis for the disc
        if abs(current_dir[0]) < 0.9:
            u = np.cross(current_dir, np.array([1, 0, 0]))
        else:
            u = np.cross(current_dir, np.array([0, 1, 0]))
        u = u / np.linalg.norm(u)
        v = np.cross(current_dir, u)

        best_pos = None
        best_clearance = -1

        # Sample on disc with varying radii (0 to MIN_CLEARANCE)
        for i in range(n_candidates):
            angle = 2 * np.pi * i / n_candidates
            for radius_frac in [0.0, 0.3, 0.6, 1.0]:
                r = MIN_CLEARANCE * radius_frac
                candidate = ahead + r * (np.cos(angle) * u + np.sin(angle) * v)
                dist, _ = tree.query(candidate)
                if dist > best_clearance:
                    best_clearance = dist
                    best_pos = candidate

        if best_clearance < MIN_CLEARANCE * 0.5:
            print(f"    Step {step}: clearance {best_clearance:.1f}A < threshold, stopping")
            break

        if best_clearance > EXIT_THRESHOLD:
            # Exited the tunnel — record exit point and extend
            print(f"    Step {step}: clearance {best_clearance:.1f}A > {EXIT_THRESHOLD}A, "
                  f"tunnel exit reached")
            centerline.append(best_pos)

            # Compute arc length up to exit point
            cl_arr = np.array(centerline)
            exit_arc_length = sum(np.linalg.norm(cl_arr[i+1] - cl_arr[i])
                                  for i in range(len(cl_arr) - 1))

            # Extend 300A past tunnel exit (150 steps * 2A)
            for ext in range(1, N_EXTENSION_STEPS + 1):
                ext_pos = best_pos + current_dir * TRACE_STEP * ext
                centerline.append(ext_pos)
            break

        centerline.append(best_pos)
        new_dir = best_pos - current_pos
        new_dir = new_dir / np.linalg.norm(new_dir)
        # Smooth direction update (momentum)
        current_dir = 0.7 * current_dir + 0.3 * new_dir
        current_dir = current_dir / np.linalg.norm(current_dir)
        current_pos = best_pos

        if step % 20 == 0:
            print(f"    Step {step}: clearance={best_clearance:.1f}A, "
                  f"pos=({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")

    centerline = np.array(centerline)
    total_length = sum(np.linalg.norm(centerline[i+1] - centerline[i])
                       for i in range(len(centerline) - 1))
    print(f"  Tunnel centerline: {len(centerline)} points, {total_length:.1f}A total length")
    if exit_arc_length is not None:
        print(f"  Tunnel exit at arc length: {exit_arc_length:.1f}A")
    return centerline, exit_arc_length


def smooth_centerline(centerline, exit_arc_length):
    """Smooth the centerline with cubic spline and variable-rate resampling.

    Inside the tunnel (before exit_arc_length): resample at EXTENDED_RISE_PER_RESIDUE
    (3.3 A) for extended conformation.
    After the tunnel exit: resample at HELIX_RISE_PER_RESIDUE (1.5 A) for alpha helix.

    Returns: (spline_points, tunnel_exit_residue, spline_fn, total_len)
    """
    # Parameterize by arc length
    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(centerline))
    t[1:] = np.cumsum(seg_lengths)
    total_len = t[-1]

    # Fit cubic spline
    cs = CubicSpline(t, centerline)

    # Build variable-rate sample points
    if exit_arc_length is None:
        # No exit found — all extended conformation
        exit_arc_length = total_len

    # Tunnel portion: extended conformation (3.3 A per residue)
    tunnel_t = np.arange(0, exit_arc_length, EXTENDED_RISE_PER_RESIDUE)
    tunnel_exit_residue = len(tunnel_t)

    # Post-exit portion: alpha helix (1.5 A per residue)
    remaining = total_len - exit_arc_length
    if remaining > 0:
        post_exit_t = np.arange(
            exit_arc_length, total_len, HELIX_RISE_PER_RESIDUE)
        # Don't duplicate the exit point
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

    Uses extended conformation geometry for residues inside the tunnel
    (before tunnel_exit_residue) and alpha helix geometry after the exit.

    Extended conformation: flatter offsets, 180 deg twist per residue
    Alpha helix: standard helix offsets, 100 deg twist per residue

    Returns: AtomArray
    """
    n_res = len(spline_points)
    print(f"  Building {n_res}-residue polyalanine along tunnel spline...")
    print(f"    Residues 1-{tunnel_exit_residue}: extended conformation (tunnel)")
    print(f"    Residues {tunnel_exit_residue+1}-{n_res}: alpha helix (post-exit)")

    # Compute local coordinate frames along the spline
    tangents = np.zeros_like(spline_points)
    tangents[0] = spline_points[1] - spline_points[0]
    tangents[-1] = spline_points[-1] - spline_points[-2]
    for i in range(1, n_res - 1):
        tangents[i] = spline_points[i + 1] - spline_points[i - 1]
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    atoms_per_res = 5  # N, CA, C, O, CB
    total_atoms = n_res * atoms_per_res
    arr = AtomArray(total_atoms)

    # Extended conformation offsets — flatter, more elongated
    # Diameter ~6 A (fits ~10 A L4-L22 constriction with clearance)
    offsets_extended = {
        "N":  np.array([-0.40, -0.30, -1.65]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C":  np.array([0.40, 0.30, 1.65]),
        "O":  np.array([0.20, 1.20, 2.00]),
        "CB": np.array([-1.52, 0.0, 0.22]),
    }

    # Alpha helix offsets — standard helix geometry
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

        # Build local frame
        if abs(t[0]) < 0.9:
            n_vec = np.cross(t, np.array([1, 0, 0]))
        else:
            n_vec = np.cross(t, np.array([0, 1, 0]))
        n_vec = n_vec / np.linalg.norm(n_vec)
        b_vec = np.cross(t, n_vec)

        # Switch geometry based on tunnel position
        if i < tunnel_exit_residue:
            # Extended conformation: 180 deg twist per residue
            offsets = offsets_extended
            twist_angle = np.radians(180) * i
        else:
            # Alpha helix: 100 deg twist per residue
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


def declash_structure(coords, ribosome_tree, ribo_coords, min_dist=3.0, max_iter=100):
    """Push polypeptide atoms away from ribosome walls.

    Iteratively displaces any atom closer than min_dist to the nearest
    ribosome atom, pushing it radially outward until clearance is achieved.

    Returns: modified coords (in-place).
    """
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
        norms = np.maximum(norms, 1e-6)  # avoid division by zero
        push_dir = push_dir / norms
        deficit = min_dist - dists[clashing]
        coords[clashing] += push_dir * deficit[:, None]
        if iteration % 20 == 0:
            print(f"    Iteration {iteration}: {n_clash} clashing atoms, "
                  f"min dist={dists.min():.2f}A")
    return coords


def verify_clearance(label, coords, ribosome_tree):
    """Verify and print clearance stats for ALL atoms against ribosome."""
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
    """3-stage constrained MD with wall repulsion.

    Phase 1: 50K steps at 400K (k_restraint=50, k_wall=1000)
    Phase 2: 30K steps at 350K (k_restraint=20, k_wall=3000)
    Phase 3: 20K steps at 310K (k_restraint=10, k_wall=5000)
    Final energy minimization.
    """
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

    # Load polypeptide
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

    # Position restraints (ramped via global parameter)
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

    # Wall repulsion force: penalizes atoms closer than 3 A to nearest ribosome atom.
    # Pre-compute nearest ribosome wall point per peptide atom.
    ribo_tree = KDTree(ribo_coords)
    pep_coords_nm = np.array([positions[i].value_in_unit(nanometer)
                               for i in range(n_peptide_atoms)])
    pep_coords_A = pep_coords_nm * 10.0  # nm -> Angstroms for KDTree query

    _, nearest_idx = ribo_tree.query(pep_coords_A)
    nearest_ribo_A = ribo_coords[nearest_idx]  # in Angstroms
    nearest_ribo_nm = nearest_ribo_A * 0.1  # -> nm

    # Wall force: quadratic repulsion from pre-computed wall points
    # r_min = 0.3 nm (3 A), distance measured from wall point
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
    # Force CPU platform (OpenCL can hang on Apple Silicon)
    from openmm import Platform
    platform = Platform.getPlatformByName('CPU')
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Initial minimize
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=0)

    # Phase 1: 400K, k_restraint=50, k_wall=1000
    print(f"  Phase 1: {PHASE1_STEPS} steps at {PHASE1_TEMP}K "
          f"(k_restraint=50, k_wall=1000)...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(PHASE1_STEPS // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(PHASE1_STEPS)

    # Phase 2: 350K, k_restraint=20, k_wall=3000
    print(f"  Phase 2: {PHASE2_STEPS} steps at {PHASE2_TEMP}K "
          f"(k_restraint=20, k_wall=3000)...")
    integrator.setTemperature(PHASE2_TEMP * kelvin)
    sim.context.setParameter("k_restraint", 20.0)
    sim.context.setParameter("k_wall", 3000.0)
    sim.step(PHASE2_STEPS)

    # Phase 3: 310K, k_restraint=10, k_wall=5000
    print(f"  Phase 3: {PHASE3_STEPS} steps at {PHASE3_TEMP}K "
          f"(k_restraint=10, k_wall=5000)...")
    integrator.setTemperature(PHASE3_TEMP * kelvin)
    sim.context.setParameter("k_restraint", 10.0)
    sim.context.setParameter("k_wall", 5000.0)
    sim.step(PHASE3_STEPS)

    # Final minimize
    print("  Final minimization...")
    sim.minimizeEnergy(maxIterations=0)
    state_final = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"  Final energy: {state_final.getPotentialEnergy()}")

    with open(output_pdb, "w") as f:
        OmmPDB.writeFile(sim.topology, state_final.getPositions(), f, keepIds=True)
    print(f"  Written: {output_pdb}")


def main():
    print("=== Building tunnel-threaded polypeptide ===")

    # Load ribosome
    atoms_60s, c4, full_arr = load_ribosome()
    ribo_coords = atoms_60s.coord
    ribo_tree = KDTree(ribo_coords)

    # Find PTC position
    ptc_pos = find_ptc_position(c4)

    # Get the initial direction from C4 backbone
    ca_mask = c4.atom_name == "CA"
    ca_coords = c4.coord[ca_mask]
    ca_res = c4.res_id[ca_mask]
    if len(ca_coords) >= 2:
        sort_idx = np.argsort(ca_res)
        initial_dir = ca_coords[sort_idx[0]] - ca_coords[sort_idx[-1]]
        initial_dir = initial_dir / np.linalg.norm(initial_dir)
        print(f"  Initial direction from C4: ({initial_dir[0]:.2f}, "
              f"{initial_dir[1]:.2f}, {initial_dir[2]:.2f})")
    else:
        initial_dir = None

    # Trace tunnel (now returns exit arc length)
    centerline, exit_arc_length = trace_tunnel(ribo_coords, ptc_pos, initial_dir)

    # Smooth centerline with variable-rate resampling
    spline_points, tunnel_exit_residue, spline_fn, total_len = smooth_centerline(
        centerline, exit_arc_length)

    # Build polypeptide with geometry switching
    polypeptide = build_backbone_along_spline(spline_points, tunnel_exit_residue)

    # Geometric de-clash against ribosome walls
    verify_clearance("before de-clash", polypeptide.coord, ribo_tree)
    polypeptide.coord = declash_structure(
        polypeptide.coord, ribo_tree, ribo_coords, min_dist=3.0)
    verify_clearance("after de-clash", polypeptide.coord, ribo_tree)

    # Write raw PDB
    raw_pdb = "tunnel_polypeptide_raw.pdb"
    pdb = PDBFile()
    pdb.set_structure(polypeptide)
    pdb.write(raw_pdb)
    print(f"  Raw polypeptide: {raw_pdb} ({len(polypeptide)} atoms, "
          f"{len(np.unique(polypeptide.res_id))} residues)")

    # Relax with 3-stage MD + wall repulsion (fallback to raw if it fails)
    try:
        relax_polypeptide(raw_pdb, ribo_coords, OUTPUT)
    except Exception as e:
        print(f"  WARNING: MD relaxation failed ({e}), using raw structure")
        import shutil
        shutil.copy(raw_pdb, OUTPUT)

    # Reload and verify final clearance
    final_pdb = PDBFile.read(OUTPUT)
    final_arr = final_pdb.get_structure()
    if isinstance(final_arr, AtomArrayStack):
        final_arr = final_arr[0]

    verify_clearance("final structure", final_arr.coord, ribo_tree)

    # Compute extent
    ca_coords = final_arr.coord[final_arr.atom_name == "CA"]
    if len(ca_coords) > 0:
        extent = np.linalg.norm(ca_coords[-1] - ca_coords[0])
        print(f"\n  Final polypeptide: {len(final_arr)} atoms, "
              f"{len(np.unique(final_arr.res_id))} residues")
        print(f"  CA extent: {extent:.1f}A = {extent * 0.1:.1f} BU")

    # Clean up temp file
    if os.path.exists(raw_pdb):
        os.unlink(raw_pdb)

    print("=== Done ===")


if __name__ == "__main__":
    main()
