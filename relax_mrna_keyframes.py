"""Generate MD-relaxed mRNA keyframes at different codon shift positions.

Creates keyframes at evenly-spaced codon shifts by shifting the mRNA structure
and running short MD relaxation with ribosome wall repulsion. The relaxed
per-residue centroids are saved to mrna_keyframes.npz for use by animate.py.

For shift=0, uses the existing extended_mrna.pdb centroids directly (no MD needed).
For other shifts, shifts atoms then runs MD relaxation with wall repulsion.

Output: mrna_keyframes.npz
  - shifts: array of codon shift positions
  - centroids_N: (n_residues, 3) in Angstroms for each shift N
  - residue_ids: (n_residues,) PDB residue IDs
  - codon_shift_angstrom: (3,) shift vector per codon

Run with: python3.11 relax_mrna_keyframes.py [--skip-minimize] [--step=N]
  --skip-minimize: skip MD, just shift coordinates (fast test)
  --step=N: keyframe every N codons (default: 5)
"""

import numpy as np
import os
import sys
import tempfile
import time

MRNA_PDB = "extended_mrna.pdb"
OUTPUT = "mrna_keyframes.npz"
SKIP_MINIMIZE = "--skip-minimize" in sys.argv
N_CYCLES = 38  # total codon shifts in animation

# Parse --step=N (default: every 5 codons → 9 keyframes)
STEP = 5
for arg in sys.argv:
    if arg.startswith("--step="):
        STEP = int(arg.split("=", 1)[1])

# Build keyframe shift positions: [0, step, 2*step, ..., N_CYCLES]
KEYFRAME_SHIFTS = list(range(0, N_CYCLES, STEP))
if KEYFRAME_SHIFTS[-1] != N_CYCLES:
    KEYFRAME_SHIFTS.append(N_CYCLES)
print(f"Keyframe shifts: {KEYFRAME_SHIFTS} ({len(KEYFRAME_SHIFTS)} keyframes, step={STEP})")

# All ribosome chain IDs (40S + 60S)
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


def compute_codon_shift(mrna_arr):
    """Compute per-codon shift vector in Angstrom space from P-atom positions."""
    p_mask = mrna_arr.atom_name == "P"
    p_coords = mrna_arr.coord[p_mask]
    p_res_ids = mrna_arr.res_id[p_mask]

    unique_res = np.sort(np.unique(p_res_ids))
    centroids = np.array([p_coords[p_res_ids == r].mean(axis=0) for r in unique_res])

    # Principal axis via SVD
    centered = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]

    # Per-nucleotide spacing along axis, x3 for one codon
    projections = centroids @ axis
    spacings = np.diff(np.sort(projections))
    nt_spacing = np.median(spacings)
    codon_shift = 3.0 * nt_spacing * axis

    # Ensure shift direction is along increasing residue index
    first_proj = centroids[0] @ axis
    last_proj = centroids[-1] @ axis
    if last_proj < first_proj:
        codon_shift = -codon_shift

    print(f"  Codon shift (Angstrom): magnitude={np.linalg.norm(codon_shift):.2f} A")
    print(f"  Direction: ({codon_shift[0]:.2f}, {codon_shift[1]:.2f}, {codon_shift[2]:.2f})")
    return codon_shift


def compute_residue_centroids(arr):
    """Compute per-residue centroids from atom array. Returns (centroids, residue_ids)."""
    unique_res = np.sort(np.unique(arr.res_id))
    centroids = np.zeros((len(unique_res), 3))
    for i, rid in enumerate(unique_res):
        centroids[i] = arr.coord[arr.res_id == rid].mean(axis=0)
    return centroids, unique_res


def load_all_ribosome_atoms():
    """Load ALL ribosome atoms from 6Y0G. Returns (all_coords, KDTree)."""
    from scipy.spatial import KDTree
    import molecularnodes as mn
    import bpy

    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    mol = mn.Molecule.fetch("6Y0G")
    arr = mol.array
    from biotite.structure import AtomArrayStack
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    mask_ribo = np.isin(arr.chain_id, RIBOSOME_CHAINS)
    all_coords = arr[mask_ribo].coord
    print(f"  All ribosome atoms: {len(all_coords)}")
    return all_coords, KDTree(all_coords)


def get_nearby_ribosome_atoms(all_ribo_coords, all_ribo_tree, mrna_coords, cutoff=15.0):
    """Get ribosome atoms within cutoff of mRNA atoms."""
    from scipy.spatial import KDTree
    mrna_tree = KDTree(mrna_coords)
    dists, _ = mrna_tree.query(all_ribo_coords)
    nearby = all_ribo_coords[dists < cutoff]
    print(f"  Ribosome atoms within {cutoff}A of mRNA: {len(nearby)}")
    return nearby


def _update_wall_anchors(wall_force, sim, all_ribo_tree, all_ribo_coords):
    """Recompute nearest ribosome atoms and update wall force anchor positions.

    Called periodically during MD to keep wall repulsion tracking the mRNA's
    current position (not its starting position).
    """
    from openmm.unit import nanometer, angstrom
    state = sim.context.getState(getPositions=True)
    pos_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    _, nearest_idx = all_ribo_tree.query(pos_A)
    nearest_nm = all_ribo_coords[nearest_idx] * 0.1  # Å → nm

    for i in range(wall_force.getNumParticles()):
        wall_force.setParticleParameters(i, i, [
            nearest_nm[i, 0], nearest_nm[i, 1], nearest_nm[i, 2]])
    wall_force.updateParametersInContext(sim.context)


def relax_shifted_mrna(mrna_arr, shift_vec, ribo_coords,
                       all_ribo_coords=None, all_ribo_tree=None,
                       k_restraint=25.0, n_steps=100000):
    """Run MD relaxation on shifted mRNA with adaptive wall repulsion.

    Args:
        mrna_arr: biotite atom array (will be shifted in-place)
        shift_vec: (3,) shift to apply before relaxation
        ribo_coords: (N, 3) nearby ribosome atom coords in Angstroms (for initial wall setup)
        all_ribo_coords: (M, 3) ALL ribosome atom coords for adaptive wall anchor updates
        all_ribo_tree: KDTree of all_ribo_coords
        k_restraint: position restraint spring constant (kJ/mol/nm^2)
        n_steps: total MD steps
    """
    from openmm.app import (
        PDBFile as OmmPDB, ForceField, Modeller, Simulation,
        CutoffNonPeriodic, HBonds,
    )
    from openmm import (
        LangevinMiddleIntegrator, CustomExternalForce, Platform,
    )
    from openmm.unit import kelvin, picosecond, picoseconds, nanometer
    from scipy.spatial import KDTree
    from biotite.structure.io.pdb import PDBFile

    WALL_UPDATE_INTERVAL = 5000  # update wall anchors every N steps (frequent to prevent drift at 400K)

    # Shift coordinates
    mrna_arr.coord += shift_vec

    # Write shifted PDB (strip CONECT)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    pdb_out = PDBFile()
    pdb_out.set_structure(mrna_arr)
    pdb_out.write(tmp.name)
    tmp.close()

    # Re-read stripping CONECT
    with open(tmp.name) as f:
        lines = [line for line in f if not line.startswith("CONECT")]
    clean = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    clean.writelines(lines)
    clean.close()
    os.unlink(tmp.name)

    try:
        pdb = OmmPDB(clean.name)
    finally:
        os.unlink(clean.name)

    ff = ForceField("amber14-all.xml")
    modeller = Modeller(pdb.topology, pdb.positions)

    # Fix RNA 5' terminus
    first_res = list(modeller.topology.residues())[0]
    to_remove = [a for a in first_res.atoms()
                 if a.name in ("P", "OP1", "OP2", "OP3")]
    if to_remove:
        modeller.delete(to_remove)
    modeller.addHydrogens(ff)

    n_atoms = modeller.topology.getNumAtoms()
    print(f"    {n_atoms} atoms (with H)")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=CutoffNonPeriodic,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )

    # Position restraints (gentle — allow conformational change near walls)
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

    # Wall repulsion from ribosome (use all_ribo for KDTree queries)
    query_ribo = all_ribo_coords if all_ribo_coords is not None else ribo_coords
    wall_force = None
    if query_ribo is not None and len(query_ribo) > 0:
        query_tree = all_ribo_tree if all_ribo_tree is not None else KDTree(query_ribo)
        mol_coords_nm = np.array([positions[i].value_in_unit(nanometer)
                                   for i in range(n_atoms)])
        mol_coords_A = mol_coords_nm * 10.0

        _, nearest_idx = query_tree.query(mol_coords_A)
        nearest_ribo_nm = query_ribo[nearest_idx] * 0.1

        wall_force = CustomExternalForce(
            "0.5*k_wall*step(r_min-dist)*(min(r_min-dist, cap)^2);"
            "dist=sqrt((x-wx)^2+(y-wy)^2+(z-wz)^2);"
            "r_min=0.5; cap=0.3"
        )
        wall_force.addGlobalParameter("k_wall", 1000.0)
        wall_force.addPerParticleParameter("wx")
        wall_force.addPerParticleParameter("wy")
        wall_force.addPerParticleParameter("wz")

        for i in range(n_atoms):
            wall_force.addParticle(i, [
                nearest_ribo_nm[i, 0], nearest_ribo_nm[i, 1],
                nearest_ribo_nm[i, 2],
            ])
        system.addForce(wall_force)
        print(f"    Wall repulsion: {len(query_ribo)} ribosome atoms for queries, "
              f"r_min=5A, k=5000, update every {WALL_UPDATE_INTERVAL} steps")

    # Backbone straightening (P-P-P angle restraints)
    from openmm import CustomAngleForce
    p_indices = [a.index for a in modeller.topology.atoms() if a.name == 'P']
    if len(p_indices) >= 3:
        angle_force = CustomAngleForce("0.5*k_angle*(theta-theta0)^2")
        angle_force.addGlobalParameter("k_angle", 50.0)
        angle_force.addGlobalParameter("theta0", np.pi)
        for i in range(len(p_indices) - 2):
            angle_force.addAngle(p_indices[i], p_indices[i + 1], p_indices[i + 2], [])
        system.addForce(angle_force)

    # Helper: run MD in chunks with periodic wall anchor updates
    def run_phase(sim, total_steps, temp, label):
        """Run MD in chunks, updating wall anchors periodically."""
        print(f"    {label}: {total_steps} MD steps (T={temp}K, "
              f"wall update every {WALL_UPDATE_INTERVAL})...")
        integrator.setTemperature(temp * kelvin)
        steps_done = 0
        while steps_done < total_steps:
            chunk = min(WALL_UPDATE_INTERVAL, total_steps - steps_done)
            sim.step(chunk)
            steps_done += chunk
            if wall_force is not None and all_ribo_tree is not None and steps_done < total_steps:
                _update_wall_anchors(wall_force, sim, all_ribo_tree, query_ribo)
                # Minimization + velocity reset to absorb new wall forces safely
                sim.minimizeEnergy(maxIterations=200)
                sim.context.setVelocitiesToTemperature(temp * kelvin)
                state = sim.context.getState(getEnergy=True)
                print(f"      Step {steps_done}: E={state.getPotentialEnergy()}, "
                      f"walls updated + minimized + velocities reset")

    platform = Platform.getPlatformByName('CPU')
    integrator = LangevinMiddleIntegrator(
        400 * kelvin, 1 / picosecond, 0.002 * picoseconds)
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    print(f"    Minimizing...")
    sim.minimizeEnergy(maxIterations=500)

    phase1 = n_steps // 2
    phase2 = n_steps // 4
    phase3 = n_steps - phase1 - phase2

    run_phase(sim, phase1, 400, "Phase 1")
    run_phase(sim, phase2, 350, "Phase 2")
    run_phase(sim, phase3, 310, "Phase 3")

    print(f"    Final minimization...")
    sim.minimizeEnergy(maxIterations=500)

    # Extract final coordinates
    from openmm.unit import angstrom
    state = sim.context.getState(getPositions=True)
    final_pos_A = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    # Map back to per-residue centroids using topology
    residue_centroids = {}
    for residue in sim.topology.residues():
        atom_indices = [a.index for a in residue.atoms()]
        residue_centroids[residue.index] = final_pos_A[atom_indices].mean(axis=0)

    # Sort by residue index
    sorted_res = sorted(residue_centroids.keys())
    centroids = np.array([residue_centroids[r] for r in sorted_res])

    return centroids, np.array(sorted_res)


def main():
    t_start = time.time()

    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile

    print("=== Generating mRNA keyframes ===")

    # Load base mRNA
    pdb = PDBFile.read(MRNA_PDB)
    base_arr = pdb.get_structure(model=1)
    if isinstance(base_arr, AtomArrayStack):
        base_arr = base_arr[0]
    print(f"  Loaded {MRNA_PDB}: {len(base_arr)} atoms, "
          f"{len(np.unique(base_arr.res_id))} residues")

    # Compute codon shift vector
    codon_shift = compute_codon_shift(base_arr)

    # Compute base centroids (shift=0, no MD needed)
    centroids_0, res_ids = compute_residue_centroids(base_arr)
    print(f"  Shift 0: {len(centroids_0)} residue centroids (from existing PDB)")

    if SKIP_MINIMIZE:
        # Just shift without MD (for quick testing)
        all_centroids = {'centroids_0': centroids_0}
        for shift in KEYFRAME_SHIFTS[1:]:
            arr_copy = base_arr.copy()
            arr_copy.coord += shift * codon_shift
            c, _ = compute_residue_centroids(arr_copy)
            all_centroids[f'centroids_{shift}'] = c
            print(f"  Shift {shift}: {len(c)} centroids (no MD, --skip-minimize)")
    else:
        # Load ALL ribosome atoms once (for adaptive wall anchor KDTree queries)
        print("  Loading ribosome context...")
        all_ribo_coords, all_ribo_tree = load_all_ribosome_atoms()

        # Get nearby subset for initial wall setup
        from scipy.spatial import KDTree as _KDTree
        mrna_tree = _KDTree(base_arr.coord)
        dists, _ = mrna_tree.query(all_ribo_coords)
        nearby_ribo = all_ribo_coords[dists < 15.0]
        print(f"  Nearby ribosome atoms (15A): {len(nearby_ribo)}")

        all_centroids = {'centroids_0': centroids_0}
        for shift in KEYFRAME_SHIFTS[1:]:
            print(f"\n--- Keyframe: shift={shift} codons ---")
            shift_vec = shift * codon_shift
            print(f"  Shift magnitude: {np.linalg.norm(shift_vec):.1f} A")

            arr_copy = base_arr.copy()

            t_kf = time.time()
            centroids, md_res_ids = relax_shifted_mrna(
                arr_copy, shift_vec, nearby_ribo,
                all_ribo_coords=all_ribo_coords,
                all_ribo_tree=all_ribo_tree,
                k_restraint=25.0, n_steps=100000)

            # The MD residue IDs are 0-based (OpenMM topology).
            # We need to store centroids in the same order as res_ids from base.
            # Since both are sequential and same count, they match ordinally.
            all_centroids[f'centroids_{shift}'] = centroids
            dt = time.time() - t_kf
            print(f"  Relaxation: {dt:.1f}s")

    # Save keyframes
    np.savez(OUTPUT,
             shifts=np.array(KEYFRAME_SHIFTS),
             residue_ids=res_ids,
             codon_shift_angstrom=codon_shift,
             **all_centroids)

    dt_total = time.time() - t_start
    print(f"\n=== Done! Saved {OUTPUT} ({len(KEYFRAME_SHIFTS)} keyframes, "
          f"{len(res_ids)} residues) in {dt_total:.1f}s ===")


if __name__ == "__main__":
    main()
