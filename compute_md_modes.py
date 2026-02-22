"""Compute PCA structural modes from MD trajectories for mRNA and tRNA.

Runs extended MD simulation, collects per-residue centroid snapshots,
and extracts principal components of motion via SVD. The resulting modes
are displacement vectors that can be modulated with integer-harmonic
sines for physically realistic, seamlessly looping structural deformation.

Output:
  mrna_modes.npz  — top 8 modes for mRNA (n_residues, 3) each
  trna_modes.npz  — top 8 modes for tRNA (n_residues, 3) each

Run with: python3.11 compute_md_modes.py
"""

import numpy as np
import sys
import os
import tempfile

N_MODES = 8
MD_STEPS = 200000
SAVE_INTERVAL = 500  # save every 500 steps = 400 snapshots
TEMPERATURE = 300  # K
OUTPUT_MRNA = "mrna_modes.npz"
OUTPUT_TRNA = "trna_modes.npz"


def collect_md_snapshots(input_pdb, md_steps, save_interval, temperature,
                         molecule_type="rna"):
    """Run MD and collect per-residue centroid snapshots.

    Returns:
        centroids: (n_snapshots, n_residues, 3) array
        residue_ids: (n_residues,) unique residue IDs
    """
    from openmm.app import (
        PDBFile as OmmPDB, ForceField, Modeller, Simulation,
        NoCutoff, HBonds, StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator
    from openmm.unit import kelvin, picosecond, picoseconds, nanometer, angstrom

    print(f"  Loading {input_pdb}...")

    # Strip CONECT records
    with open(input_pdb) as f:
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

    if molecule_type == "rna_tiled":
        # Remove P/OP1/OP2 from 5' terminus (only for tiled mRNA, not intact tRNA)
        first_res = list(modeller.topology.residues())[0]
        to_remove = [a for a in first_res.atoms() if a.name in ("P", "OP1", "OP2")]
        if to_remove:
            print(f"  Removing {[a.name for a in to_remove]} from 5' residue")
            modeller.delete(to_remove)

    try:
        modeller.addHydrogens(ff)
    except ValueError as e:
        print(f"  WARNING: addHydrogens failed ({e})")
        print(f"  Proceeding without hydrogens...")
        # Re-create modeller without H
        modeller = Modeller(pdb.topology, pdb.positions)
        if molecule_type == "rna_tiled":
            first_res = list(modeller.topology.residues())[0]
            to_remove = [a for a in first_res.atoms() if a.name in ("P", "OP1", "OP2")]
            if to_remove:
                modeller.delete(to_remove)

    print(f"  Prepared: {modeller.topology.getNumAtoms()} atoms, "
          f"{modeller.topology.getNumResidues()} residues")

    try:
        system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff,
                                 constraints=HBonds)
    except ValueError as e:
        print(f"  WARNING: createSystem failed ({e})")
        print(f"  Trying without constraints...")
        try:
            system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff)
        except ValueError as e2:
            print(f"  ERROR: Cannot create system: {e2}")
            return None, None

    integrator = LangevinMiddleIntegrator(
        temperature * kelvin, 1 / picosecond, 0.002 * picoseconds)
    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)

    # Minimize first
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=0)

    # Build residue → atom index mapping from topology
    residue_atoms = {}
    for residue in sim.topology.residues():
        atom_indices = [a.index for a in residue.atoms()]
        residue_atoms[residue.index] = atom_indices

    residue_ids = sorted(residue_atoms.keys())
    n_residues = len(residue_ids)
    n_snapshots = md_steps // save_interval

    print(f"  Running {md_steps} MD steps, saving every {save_interval} "
          f"({n_snapshots} snapshots, {n_residues} residues)...")

    sim.reporters.append(
        StateDataReporter(sys.stdout, max(md_steps // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )

    centroids = np.zeros((n_snapshots, n_residues, 3))

    for snap_idx in range(n_snapshots):
        sim.step(save_interval)
        state = sim.context.getState(getPositions=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(angstrom)

        for res_idx, res_id in enumerate(residue_ids):
            atom_idx = residue_atoms[res_id]
            centroids[snap_idx, res_idx] = positions[atom_idx].mean(axis=0)

    print(f"  Collected {n_snapshots} snapshots")
    return centroids, np.array(residue_ids)


def compute_pca_modes(centroids, n_modes):
    """Compute top PCA modes from centroid trajectory.

    Args:
        centroids: (n_snapshots, n_residues, 3) array
        n_modes: number of modes to extract

    Returns:
        modes: (n_modes, n_residues, 3) displacement vectors
        variance_explained: (n_modes,) fraction of variance per mode
    """
    n_snap, n_res, _ = centroids.shape

    # Compute mean structure
    mean_struct = centroids.mean(axis=0)  # (n_res, 3)

    # Deviations from mean, flattened to (n_snap, n_res*3)
    deviations = (centroids - mean_struct).reshape(n_snap, -1)

    # SVD (no need for ProDy — numpy SVD is sufficient)
    U, S, Vt = np.linalg.svd(deviations, full_matrices=False)

    # Variance explained
    total_var = (S ** 2).sum()
    variance_explained = (S[:n_modes] ** 2) / total_var

    # Extract top modes and reshape to (n_modes, n_res, 3)
    modes = Vt[:n_modes].reshape(n_modes, n_res, 3)

    # Normalize each mode to unit RMS displacement
    for i in range(n_modes):
        rms = np.sqrt((modes[i] ** 2).mean())
        if rms > 0:
            modes[i] /= rms

    print(f"  Top {n_modes} modes capture {variance_explained.sum():.1%} of variance")
    for i in range(n_modes):
        print(f"    Mode {i}: {variance_explained[i]:.1%}")

    return modes, variance_explained


def compute_modes_for_molecule(input_pdb, output_npz, molecule_type="rna"):
    """Full pipeline: MD → snapshots → PCA → save."""
    print(f"\n=== Computing PCA modes for {input_pdb} ===")

    result = collect_md_snapshots(
        input_pdb, MD_STEPS, SAVE_INTERVAL, TEMPERATURE,
        molecule_type=molecule_type)

    if result[0] is None:
        print(f"  WARNING: MD failed for {input_pdb}, skipping PCA")
        return

    centroids, residue_ids = result
    modes, variance = compute_pca_modes(centroids, N_MODES)

    # Save: modes (n_modes, n_res, 3), mean structure, variance
    mean_struct = centroids.mean(axis=0)
    np.savez(output_npz,
             modes=modes,
             mean_centroids=mean_struct,
             residue_ids=residue_ids,
             variance_explained=variance)
    print(f"  Saved: {output_npz} "
          f"({N_MODES} modes, {len(residue_ids)} residues)")


def extract_trna_pdb():
    """Extract tRNA chain B4 from 6Y0G to a standalone PDB for MD."""
    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile
    import molecularnodes as mn
    import bpy

    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    mol = mn.Molecule.fetch("6Y0G")
    arr = mol.array
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    b4 = arr[arr.chain_id == "B4"]
    # Remap 2-char chain ID to single char for PDB compatibility
    b4 = b4.copy()
    b4.chain_id[:] = "B"
    print(f"  Chain B4: {len(b4)} atoms, {len(np.unique(b4.res_id))} residues")

    out_path = "trna_b4.pdb"
    pdb = PDBFile()
    pdb.set_structure(b4)
    pdb.write(out_path)
    print(f"  Written: {out_path}")
    return out_path


def main():
    print("=== Computing PCA structural modes ===")

    # mRNA modes (from already-built extended_mrna.pdb)
    if os.path.exists(OUTPUT_MRNA):
        print(f"  {OUTPUT_MRNA} already exists, skipping mRNA (delete to recompute)")
    elif not os.path.exists("extended_mrna.pdb"):
        print("ERROR: extended_mrna.pdb not found. Run build_extended_mrna.py first.")
        return
    else:
        compute_modes_for_molecule("extended_mrna.pdb", OUTPUT_MRNA, molecule_type="rna_tiled")

    # tRNA modes (extract chain B4 from 6Y0G)
    trna_pdb = extract_trna_pdb()
    compute_modes_for_molecule(trna_pdb, OUTPUT_TRNA, molecule_type="rna")

    print("\n=== Done! ===")
    print(f"  {OUTPUT_MRNA}: mRNA structural modes")
    print(f"  {OUTPUT_TRNA}: tRNA structural modes")


if __name__ == "__main__":
    main()
