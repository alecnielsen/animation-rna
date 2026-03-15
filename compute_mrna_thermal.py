"""Pre-compute mRNA ENM thermal trajectory for coordinated breathing motion.

Uses coarse-grained elastic network model (one bead per residue, P-atom
positions). ENM springs connect nearby residues; Langevin dynamics at 310K
produces coordinated whole-structure undulations rather than per-atom jitter.

Output: mrna_thermal.npz
  - deltas: (n_frames, n_residues, 3) in BU (Å × 0.01)
  - residue_ids: (n_residues,) PDB residue IDs

Run with: python3.11 compute_mrna_thermal.py [--frames=N] [--amplitude=F]
"""

import numpy as np
import sys
import time

MRNA_PDB = "extended_mrna.pdb"
OUTPUT = "mrna_thermal.npz"

# Parse args
N_FRAMES = 456  # match ribosome thermal
STEPS_PER_FRAME = 500
TEMPERATURE = 310  # K
AMPLITUDE = 3.0  # multiplier on deltas for visible motion

for arg in sys.argv[1:]:
    if arg.startswith("--frames="):
        N_FRAMES = int(arg.split("=", 1)[1])
    elif arg.startswith("--amplitude="):
        AMPLITUDE = float(arg.split("=", 1)[1])


def main():
    t_start = time.time()

    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile

    print(f"=== mRNA ENM thermal motion ({N_FRAMES} frames, "
          f"{STEPS_PER_FRAME} steps/frame, T={TEMPERATURE}K, "
          f"amplitude={AMPLITUDE}x) ===")

    # Load mRNA
    pdb = PDBFile.read(MRNA_PDB)
    arr = pdb.get_structure(model=1)
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]
    print(f"  Loaded {MRNA_PDB}: {len(arr)} atoms")

    # Coarse-grain: one bead per residue using P atoms (or centroid fallback)
    unique_res = np.sort(np.unique(arr.res_id))
    n_residues = len(unique_res)
    bead_coords = np.zeros((n_residues, 3))  # Angstroms

    for i, rid in enumerate(unique_res):
        res_mask = arr.res_id == rid
        res_atoms = arr[res_mask]
        p_mask = res_atoms.atom_name == "P"
        if np.any(p_mask):
            bead_coords[i] = res_atoms.coord[p_mask][0]
        else:
            bead_coords[i] = res_atoms.coord.mean(axis=0)

    print(f"  Residues: {n_residues} beads")

    # Build ENM system
    from openmm import (
        System, LangevinMiddleIntegrator, CustomBondForce,
        CustomExternalForce, Platform, Vec3,
    )
    from openmm.app import Simulation, Topology, Element
    from openmm.unit import (
        kelvin, picosecond, picoseconds, nanometer, dalton, angstrom,
    )
    from scipy.spatial import KDTree

    # Topology: one particle per residue
    topology = Topology()
    chain_top = topology.addChain()
    carbon = Element.getBySymbol('C')
    for ri in range(n_residues):
        res = topology.addResidue(f"R{ri}", chain_top)
        topology.addAtom("CA", carbon, res)

    system = System()
    bead_coords_nm = bead_coords * 0.1  # Å → nm

    for ri in range(n_residues):
        system.addParticle(100.0 * dalton)

    # ENM spring network
    ENM_CUTOFF = 1.5  # nm (15 Å)
    ENM_K = 5.0  # kJ/mol/nm² — softer than ribosome for more visible motion

    tree = KDTree(bead_coords_nm)
    pairs = tree.query_pairs(ENM_CUTOFF)
    print(f"  ENM springs: {len(pairs)} (cutoff={ENM_CUTOFF}nm, k={ENM_K})")

    bond_force = CustomBondForce("0.5*k_enm*(r-r0)^2")
    bond_force.addGlobalParameter("k_enm", ENM_K)
    bond_force.addPerBondParameter("r0")

    for i, j in pairs:
        r0 = np.linalg.norm(bead_coords_nm[i] - bead_coords_nm[j])
        bond_force.addBond(i, j, [r0])
    system.addForce(bond_force)

    # Position restraints (softer than ribosome — mRNA is a flexible chain)
    K_RESTRAINT = 20.0  # kJ/mol/nm² (ribosome uses 100)
    restraint = CustomExternalForce(
        "0.5*k_restraint*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k_restraint", K_RESTRAINT)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for i in range(n_residues):
        restraint.addParticle(i, [
            bead_coords_nm[i, 0], bead_coords_nm[i, 1], bead_coords_nm[i, 2]
        ])
    system.addForce(restraint)
    print(f"  Position restraints: k={K_RESTRAINT} kJ/mol/nm²")

    # Langevin dynamics
    integrator = LangevinMiddleIntegrator(
        TEMPERATURE * kelvin, 1 / picosecond, 0.002 * picoseconds)

    platform = Platform.getPlatformByName('CPU')
    positions = [Vec3(bead_coords_nm[i, 0], bead_coords_nm[i, 1],
                      bead_coords_nm[i, 2]) * nanometer
                 for i in range(n_residues)]

    sim = Simulation(topology, system, integrator, platform)
    sim.context.setPositions(positions)

    # Minimize + thermalize
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=500)

    print("  Thermalizing (2000 steps)...")
    sim.step(2000)

    # Record rest positions
    state = sim.context.getState(getPositions=True)
    rest_pos = state.getPositions(asNumpy=True).value_in_unit(angstrom)

    # Run trajectory
    print(f"  Running {N_FRAMES} frames × {STEPS_PER_FRAME} steps...")
    deltas = np.zeros((N_FRAMES, n_residues, 3), dtype=np.float32)

    for fi in range(N_FRAMES):
        sim.step(STEPS_PER_FRAME)
        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True).value_in_unit(angstrom)
        # Delta in Å → BU (× 0.01), then amplify
        deltas[fi] = (pos - rest_pos) * 0.01 * AMPLITUDE

        if (fi + 1) % 50 == 0:
            rms = np.sqrt(np.mean(deltas[fi] ** 2))
            print(f"    Frame {fi + 1}/{N_FRAMES}: RMS delta = {rms:.4f} BU")

    # Cross-fade last 12 frames for seamless loop
    BLEND_N = 12
    for i in range(BLEND_N):
        t = (i + 1) / BLEND_N
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        fi = N_FRAMES - BLEND_N + i
        deltas[fi] = (1.0 - t) * deltas[fi] + t * deltas[i]

    # Save
    np.savez_compressed(OUTPUT,
                        deltas=deltas,
                        residue_ids=unique_res)

    dt = time.time() - t_start
    rms_all = np.sqrt(np.mean(deltas ** 2))
    print(f"\n=== Done! Saved {OUTPUT} ({N_FRAMES} frames, {n_residues} residues, "
          f"RMS={rms_all:.4f} BU) in {dt:.1f}s ===")


if __name__ == "__main__":
    main()
