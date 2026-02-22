"""Build an extended mRNA strand by tiling chain A4 from 6Y0G.

Creates a single continuous ~170-nucleotide mRNA from 10 copies of the
17 nt chain A4, with correct backbone spacing and sequential residue
numbering. Randomizes nucleotide sequence per tile to produce genuinely
different backbone conformations. Then runs extended 3-stage MD annealing
with OpenMM (amber14 RNA.OL3 force field) to break tile symmetry.

Protocol: 500K MD steps total
  - 0.5A random perturbation per tile before MD (seed different pathways)
  - Randomized nucleotide sequence per tile (ACGU mix)
  - 300K steps at 400K (high-temperature conformational sampling)
  - 100K steps at 350K (intermediate cooling)
  - 100K steps at 310K (physiological temperature)
  - Final energy minimization (quench)

Output: extended_mrna.pdb

Run with: python3.11 build_extended_mrna.py [--skip-minimize]
"""

import molecularnodes as mn
import bpy
import numpy as np
import sys
import os
import tempfile
from biotite.structure import AtomArrayStack, concatenate, connect_via_residue_names
from biotite.structure.io.pdb import PDBFile

N_COPIES = 10
CENTER_INDEX = 5  # copy index that stays at crystallographic position
OUTPUT = "extended_mrna.pdb"
SKIP_MINIMIZE = "--skip-minimize" in sys.argv


def tile_mrna():
    """Fetch 6Y0G chain A4, tile it N_COPIES times, write raw PDB."""
    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    print("=== Building extended mRNA ===")

    # Load 6Y0G and extract chain A4
    mol = mn.Molecule.fetch("6Y0G")
    arr = mol.array
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    a4_raw = arr[arr.chain_id == "A4"]
    # Filter out non-nucleotide heteroatoms (e.g. MG²⁺ ion at res 101)
    nuc_names = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}
    nuc_mask = np.isin(a4_raw.res_name, list(nuc_names))
    a4 = a4_raw[nuc_mask]
    print(f"  Chain A4: {len(a4_raw)} atoms total, {len(a4)} nucleotide atoms")

    # Compute tile offset using backbone atom positions for tight junctions.
    unique_res = np.unique(a4.res_id)
    o3_last = a4.coord[(a4.res_id == unique_res[-1]) & (a4.atom_name == "O3'")][0]
    p_first = a4.coord[(a4.res_id == unique_res[0]) & (a4.atom_name == "P")][0]
    o3_res0 = a4.coord[(a4.res_id == unique_res[0]) & (a4.atom_name == "O3'")][0]
    p_res1 = a4.coord[(a4.res_id == unique_res[1]) & (a4.atom_name == "P")][0]
    bond_vec = p_res1 - o3_res0  # internal O3'→P step (~1.6 Å)
    tile_offset = o3_last + bond_vec - p_first

    p_coords = a4.coord[a4.atom_name == "P"]
    print(f"  P atoms: {len(p_coords)}")
    print(f"  Tile offset: ({tile_offset[0]:.1f}, {tile_offset[1]:.1f}, {tile_offset[2]:.1f}) A")
    print(f"  |tile_offset| = {np.linalg.norm(tile_offset):.1f} A")

    # Map each atom's res_id to a 0-based index within the tile
    n_res = len(unique_res)
    res_to_idx = {int(r): j for j, r in enumerate(unique_res)}
    base_idx = np.array([res_to_idx[int(r)] for r in a4.res_id])
    print(f"  Residues per tile: {n_res}")

    # Create N_COPIES offset copies, centered so i=CENTER_INDEX is at origin
    # Randomize sequence per tile and add perturbation to seed different pathways
    rng = np.random.default_rng(42)
    nuc_choices = ["A", "C", "G", "U"]
    copies = []
    for i in range(N_COPIES):
        tile = a4.copy()
        tile.coord += (i - CENTER_INDEX) * tile_offset
        # 0.5A random perturbation per atom (was 0.2A) for stronger symmetry breaking
        tile.coord += rng.normal(0, 0.5, tile.coord.shape)
        tile.res_id = (base_idx + i * n_res + 1).astype(a4.res_id.dtype)
        tile.chain_id[:] = "A"
        # Randomize nucleotide sequence per tile — different force field
        # interactions produce genuinely different backbone conformations
        tile_unique_res = np.unique(tile.res_id)
        for res in tile_unique_res:
            new_name = rng.choice(nuc_choices)
            tile.res_name[tile.res_id == res] = new_name
        copies.append(tile)

    extended = concatenate(copies)
    print(f"  Extended: {len(extended)} atoms, "
          f"{len(np.unique(extended.res_id))} residues")

    # Generate bonds
    extended.bonds = connect_via_residue_names(extended, inter_residue=True)

    # Write raw tiled PDB
    pdb = PDBFile()
    pdb.set_structure(extended)
    pdb.write(OUTPUT)
    print(f"  Written: {OUTPUT}")

    return extended, n_res


def _load_and_prepare(input_pdb):
    """Load RNA PDB into OpenMM, fix termini, add hydrogens. Returns (ff, modeller)."""
    from openmm.app import PDBFile as OmmPDB, ForceField, Modeller

    # Strip CONECT records (biotite writes per-tile CONECTs with wrapping serials)
    with open(input_pdb) as f:
        lines = [line for line in f if not line.startswith("CONECT")]
    clean = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w")
    clean.writelines(lines)
    clean.close()

    try:
        pdb = OmmPDB(clean.name)
    finally:
        os.unlink(clean.name)

    print(f"  Loaded: {pdb.topology.getNumAtoms()} atoms, "
          f"{pdb.topology.getNumResidues()} residues")

    ff = ForceField("amber14-all.xml", "implicit/gbn2.xml")

    # Remove P/OP1/OP2 from 5' terminus (amber14 U5 template expects HO5'+O5')
    modeller = Modeller(pdb.topology, pdb.positions)
    first_res = list(modeller.topology.residues())[0]
    to_remove = [a for a in first_res.atoms() if a.name in ("P", "OP1", "OP2")]
    if to_remove:
        print(f"  Removing {[a.name for a in to_remove]} from 5' residue")
        modeller.delete(to_remove)

    modeller.addHydrogens(ff)
    print(f"  After adding H: {modeller.topology.getNumAtoms()} atoms")

    return ff, modeller


def relax_rna(input_pdb, output_pdb):
    """3-stage annealing to eliminate tile periodicity.

    Pipeline:
      1. Energy minimization
      2. 300K MD steps at 400K (high-temperature conformational sampling)
      3. 100K MD steps at 350K (intermediate cooling)
      4. 100K MD steps at 310K (physiological temperature)
      5. Final energy minimization (quench)

    Total: 500K steps. The 3-stage annealing with per-tile sequence
    randomization and 0.5A perturbation produces diverse backbone
    conformations that eliminate visible tiling patterns.
    """
    from openmm.app import (
        PDBFile as OmmPDB, Simulation, NoCutoff, HBonds,
        StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator
    from openmm.unit import kelvin, picosecond, picoseconds, nanometer

    PHASE1_STEPS = 300000
    PHASE1_TEMP = 400  # K
    PHASE2_STEPS = 100000
    PHASE2_TEMP = 350  # K
    PHASE3_STEPS = 100000
    PHASE3_TEMP = 310  # K
    TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS

    print(f"\n=== Relaxation ({TOTAL_STEPS} MD steps: "
          f"{PHASE1_STEPS}@{PHASE1_TEMP}K + {PHASE2_STEPS}@{PHASE2_TEMP}K + "
          f"{PHASE3_STEPS}@{PHASE3_TEMP}K + quench) ===")

    ff, modeller = _load_and_prepare(input_pdb)

    system = ff.createSystem(modeller.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

    # Phase 1: 400K high-temperature conformational sampling
    integrator = LangevinMiddleIntegrator(
        PHASE1_TEMP * kelvin, 1 / picosecond, 0.002 * picoseconds)
    sim = Simulation(modeller.topology, system, integrator)
    sim.context.setPositions(modeller.positions)

    # Step 1: Initial minimization
    state0 = sim.context.getState(getEnergy=True)
    print(f"  Initial energy: {state0.getPotentialEnergy()}")
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=0)
    state1 = sim.context.getState(getEnergy=True)
    print(f"  Post-minimize energy: {state1.getPotentialEnergy()}")

    # Step 2: Phase 1 — MD at 400K
    print(f"  Phase 1: {PHASE1_STEPS} MD steps (dt=2fs, T={PHASE1_TEMP}K)...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(PHASE1_STEPS // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(PHASE1_STEPS)

    # Step 3: Phase 2 — cool to 350K
    print(f"  Phase 2: {PHASE2_STEPS} MD steps (dt=2fs, T={PHASE2_TEMP}K)...")
    integrator.setTemperature(PHASE2_TEMP * kelvin)
    sim.step(PHASE2_STEPS)

    # Step 4: Phase 3 — cool to 310K (physiological)
    print(f"  Phase 3: {PHASE3_STEPS} MD steps (dt=2fs, T={PHASE3_TEMP}K)...")
    integrator.setTemperature(PHASE3_TEMP * kelvin)
    sim.step(PHASE3_STEPS)

    # Step 5: Final minimization (quench)
    print("  Final minimization (quench)...")
    sim.minimizeEnergy(maxIterations=0)
    state_final = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"  Final energy: {state_final.getPotentialEnergy()}")

    with open(output_pdb, "w") as f:
        OmmPDB.writeFile(sim.topology, state_final.getPositions(), f, keepIds=True)
    print(f"  Written: {output_pdb}")


def verify(extended, n_res):
    """Print junction distances for verification."""
    print(f"\n=== Junction verification (O3'->P at tile boundaries) ===")
    for i in range(N_COPIES - 1):
        last_res = (i + 1) * n_res
        first_res = last_res + 1
        m_o3 = (extended.res_id == last_res) & (extended.atom_name == "O3'")
        m_p = (extended.res_id == first_res) & (extended.atom_name == "P")
        if np.any(m_o3) and np.any(m_p):
            d = np.linalg.norm(extended.coord[m_p][0] - extended.coord[m_o3][0])
            print(f"  Tile {i}->{i+1} (res {last_res}->{first_res}): O3'...P = {d:.2f} A")
        else:
            avail = "O3'" if np.any(m_o3) else "no O3'"
            avail += ", P" if np.any(m_p) else ", no P"
            print(f"  Tile {i}->{i+1} (res {last_res}->{first_res}): {avail}")

    p_ext = extended.coord[extended.atom_name == "P"]
    extent = np.linalg.norm(p_ext[-1] - p_ext[0])
    print(f"\n  Total P-P extent: {extent:.1f} A = {extent * 0.1:.1f} BU")
    print(f"  Total atoms: {len(extended)}")


def main():
    extended, n_res = tile_mrna()
    verify(extended, n_res)

    if SKIP_MINIMIZE:
        print("\n  Skipping minimization (--skip-minimize)")
    else:
        relax_rna(OUTPUT, OUTPUT)

    print("=== Done ===")


if __name__ == "__main__":
    main()
