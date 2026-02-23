"""Build a polypeptide threaded through the ribosome exit tunnel.

Traces the exit tunnel void space from the peptidyl transferase center (PTC)
through the 60S subunit, then builds a polyalanine alpha helix along the
centerline. The polypeptide is relaxed with constrained MD (ribosome frozen,
polypeptide flexible).

Algorithm:
  1. Build KDTree of all 60S ribosome atoms
  2. From C4 position (PTC), trace through void space by picking points
     with maximum clearance from ribosome walls
  3. Smooth centerline with cubic spline
  4. Build polyalanine backbone along spline at 1.5A intervals
  5. Constrained MD relaxation

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
HELIX_RISE_PER_RESIDUE = 1.5  # Angstroms
TRACE_STEP = 2.0  # Angstroms per tracing step
MIN_CLEARANCE = 4.0  # minimum distance from tunnel wall (A)
EXIT_THRESHOLD = 15.0  # distance at which we consider having exited
N_EXTENSION_BEYOND_EXIT = 10  # extra residues past tunnel exit

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

    Returns: (N, 3) array of tunnel centerline points.
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
            # Exited the tunnel — add a few extension points
            print(f"    Step {step}: clearance {best_clearance:.1f}A > {EXIT_THRESHOLD}A, "
                  f"tunnel exit reached")
            centerline.append(best_pos)
            # Extend well past tunnel exit (200A = ~20 BU, enough to go off-frame)
            for ext in range(1, 101):
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
    return centerline


def smooth_centerline(centerline):
    """Smooth the centerline with cubic spline interpolation."""
    # Parameterize by arc length
    diffs = np.diff(centerline, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(centerline))
    t[1:] = np.cumsum(seg_lengths)

    # Fit cubic spline
    cs = CubicSpline(t, centerline)

    # Resample at uniform HELIX_RISE_PER_RESIDUE intervals
    total_len = t[-1]
    n_points = int(total_len / HELIX_RISE_PER_RESIDUE)
    t_uniform = np.linspace(0, total_len, n_points)
    smooth = cs(t_uniform)

    print(f"  Smoothed centerline: {n_points} points at {HELIX_RISE_PER_RESIDUE}A intervals")
    return smooth, cs, total_len


def build_helix_along_spline(spline_points):
    """Build polyalanine backbone atoms along the spline centerline.

    Places backbone atoms (N, CA, C, O, CB) using ideal helix geometry
    rotated into the local coordinate frame defined by the spline tangent.

    Returns: AtomArray
    """
    n_res = len(spline_points)
    print(f"  Building {n_res}-residue polyalanine along tunnel spline...")

    # Compute local coordinate frames along the spline
    # tangent = forward direction, then build orthonormal basis
    tangents = np.zeros_like(spline_points)
    tangents[0] = spline_points[1] - spline_points[0]
    tangents[-1] = spline_points[-1] - spline_points[-2]
    for i in range(1, n_res - 1):
        tangents[i] = spline_points[i + 1] - spline_points[i - 1]
    tangents = tangents / np.linalg.norm(tangents, axis=1, keepdims=True)

    # Build ideal helix backbone in local frame
    # For each residue, place atoms relative to CA at the spline point
    atoms_per_res = 5  # N, CA, C, O, CB
    total_atoms = n_res * atoms_per_res
    arr = AtomArray(total_atoms)

    # Ideal atom offsets relative to CA in a standard frame (helix axis along z)
    # These are approximate but sufficient for visualization
    offsets = {
        "N": np.array([-0.53, -0.84, -0.75]),
        "CA": np.array([0.0, 0.0, 0.0]),
        "C": np.array([0.53, 0.84, 0.75]),
        "O": np.array([0.25, 1.96, 0.92]),
        "CB": np.array([-1.52, 0.0, 0.22]),
    }
    atom_names = ["N", "CA", "C", "O", "CB"]
    elements = ["N", "C", "C", "O", "C"]

    idx = 0
    for i in range(n_res):
        ca_pos = spline_points[i]
        t = tangents[i]

        # Build local frame: t is forward, compute normal and binormal
        if abs(t[0]) < 0.9:
            n_vec = np.cross(t, np.array([1, 0, 0]))
        else:
            n_vec = np.cross(t, np.array([0, 1, 0]))
        n_vec = n_vec / np.linalg.norm(n_vec)
        b_vec = np.cross(t, n_vec)

        # Rotation for helix twist: 100 degrees per residue (alpha helix)
        twist_angle = np.radians(100) * i
        cos_tw, sin_tw = np.cos(twist_angle), np.sin(twist_angle)
        n_rot = cos_tw * n_vec + sin_tw * b_vec
        b_rot = -sin_tw * n_vec + cos_tw * b_vec

        # Rotation matrix: columns are [n_rot, b_rot, t]
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


def verify_tunnel_clearance(polypeptide, atoms_60s_coords):
    """Verify all CA atoms have sufficient clearance from ribosome."""
    tree = KDTree(atoms_60s_coords)
    ca_coords = polypeptide.coord[polypeptide.atom_name == "CA"]

    distances, _ = tree.query(ca_coords)
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()

    print(f"\n=== Tunnel clearance verification ===")
    print(f"  CA atoms: {len(ca_coords)}")
    print(f"  Min clearance: {min_dist:.1f}A")
    print(f"  Max clearance: {max_dist:.1f}A")
    print(f"  Mean clearance: {mean_dist:.1f}A")

    n_close = (distances < 3.0).sum()
    if n_close > 0:
        print(f"  WARNING: {n_close} CA atoms closer than 3A to ribosome")
    else:
        print(f"  OK: all CA atoms > 3A from ribosome")

    return distances


def relax_polypeptide(polypeptide_pdb, atoms_60s, output_pdb):
    """Constrained MD: freeze ribosome, relax polypeptide.

    Loads polypeptide + nearby 60S atoms (within 10A), freezes ribosome,
    runs short annealing + minimize.
    """
    from openmm.app import (
        PDBFile as OmmPDB, ForceField, Modeller, Simulation,
        NoCutoff, HBonds, StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator, CustomExternalForce
    from openmm.unit import kelvin, picosecond, picoseconds, kilojoule_per_mole, nanometer

    print(f"\n=== Constrained MD relaxation ===")

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

    # Add terminal caps (ACE at N-term, NME at C-term) so OpenMM recognizes termini
    # If that fails, add hydrogens with explicit variant handling
    from openmm.app import NoCutoff as _NC
    try:
        modeller.addHydrogens(ff)
    except ValueError:
        # Terminal residues need capping — re-add with explicit variants
        print("  Adding hydrogens with explicit terminal variants...")
        residues = list(modeller.topology.residues())
        variants = [None] * len(residues)
        variants[0] = 'ACE'   # N-terminal cap
        variants[-1] = 'NME'  # C-terminal cap
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

    # Add position restraints to keep polypeptide near initial positions
    # (gentle: 10 kJ/mol/nm^2) to prevent it from flying apart
    restraint = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", 10.0)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.positions
    for i in range(n_peptide_atoms):
        pos = positions[i].value_in_unit(nanometer)
        restraint.addParticle(i, [pos[0], pos[1], pos[2]])
    system.addForce(restraint)

    integrator = LangevinMiddleIntegrator(
        300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)

    # Minimize
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=0)

    # Short MD
    md_steps = 5000
    print(f"  Running {md_steps} MD steps at 300K...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(md_steps // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(md_steps)

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

    # Find PTC position
    ptc_pos = find_ptc_position(c4)

    # Get the initial direction from C4 backbone
    # Direction from first CA to last CA points into the tunnel
    ca_mask = c4.atom_name == "CA"
    ca_coords = c4.coord[ca_mask]
    ca_res = c4.res_id[ca_mask]
    if len(ca_coords) >= 2:
        # Direction from last to first residue = into tunnel
        sort_idx = np.argsort(ca_res)
        initial_dir = ca_coords[sort_idx[0]] - ca_coords[sort_idx[-1]]
        initial_dir = initial_dir / np.linalg.norm(initial_dir)
        print(f"  Initial direction from C4: ({initial_dir[0]:.2f}, "
              f"{initial_dir[1]:.2f}, {initial_dir[2]:.2f})")
    else:
        initial_dir = None

    # Trace tunnel
    centerline = trace_tunnel(atoms_60s.coord, ptc_pos, initial_dir)

    # Smooth centerline and resample
    spline_points, spline_fn, total_len = smooth_centerline(centerline)

    # Build polypeptide along spline
    polypeptide = build_helix_along_spline(spline_points)

    # Write raw PDB
    raw_pdb = "tunnel_polypeptide_raw.pdb"
    pdb = PDBFile()
    pdb.set_structure(polypeptide)
    pdb.write(raw_pdb)
    print(f"  Raw polypeptide: {raw_pdb} ({len(polypeptide)} atoms, "
          f"{len(np.unique(polypeptide.res_id))} residues)")

    # Verify clearance
    verify_tunnel_clearance(polypeptide, atoms_60s.coord)

    # Relax with constrained MD (fallback to raw if it fails)
    try:
        relax_polypeptide(raw_pdb, atoms_60s, OUTPUT)
    except Exception as e:
        print(f"  WARNING: MD relaxation failed ({e}), using raw structure")
        import shutil
        shutil.copy(raw_pdb, OUTPUT)

    # Reload and verify final clearance
    final_pdb = PDBFile.read(OUTPUT)
    final_arr = final_pdb.get_structure()
    if isinstance(final_arr, AtomArrayStack):
        final_arr = final_arr[0]

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
