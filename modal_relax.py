"""GPU-accelerated mRNA MD relaxation via Modal.

Runs the 500K-step 3-stage annealing protocol on a T4 GPU, which is
~50-100x faster than CPU (~1-2 min vs ~50 min).

The Modal function is fully self-contained: it fetches 6Y0G from RCSB
for ribosome wall context, so the only input needed is the raw tiled PDB.

Prerequisites:
  pip install modal
  modal setup  # one-time auth

Usage:
  # 1. Build raw tiled mRNA locally (fast, ~30s)
  source mn_env/bin/activate
  python3.11 build_extended_mrna.py --skip-minimize

  # 2. Relax on GPU via Modal (~1-2 min)
  modal run modal_relax.py

Output: extended_mrna.pdb (overwritten with relaxed structure)
"""

import modal

image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("openmm", "cudatoolkit", channels=["conda-forge"])
    .pip_install("biotite", "scipy", "numpy")
)

app = modal.App("mrna-relaxation", image=image)

# All ribosome chain IDs (40S + 60S) for wall detection
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


def _load_and_prepare(input_pdb):
    """Load RNA PDB into OpenMM, fix termini, add hydrogens."""
    import tempfile
    import os
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


def _relax_rna(input_pdb, output_pdb, ribo_coords=None):
    """3-stage annealing with wall repulsion — GPU-accelerated."""
    import sys
    import numpy as np
    from scipy.spatial import KDTree
    from openmm.app import (
        PDBFile as OmmPDB, Simulation, CutoffNonPeriodic, HBonds,
        StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator, CustomExternalForce, Platform
    from openmm.unit import kelvin, picosecond, picoseconds, nanometer

    PHASE1_STEPS = 300000
    PHASE1_TEMP = 400
    PHASE2_STEPS = 100000
    PHASE2_TEMP = 350
    PHASE3_STEPS = 100000
    PHASE3_TEMP = 310
    TOTAL = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS

    print(f"\n=== Relaxation ({TOTAL} steps: "
          f"{PHASE1_STEPS}@{PHASE1_TEMP}K + {PHASE2_STEPS}@{PHASE2_TEMP}K + "
          f"{PHASE3_STEPS}@{PHASE3_TEMP}K + quench) ===")

    ff, modeller = _load_and_prepare(input_pdb)

    system = ff.createSystem(modeller.topology, nonbondedMethod=CutoffNonPeriodic,
                             nonbondedCutoff=1.0 * nanometer, constraints=HBonds)

    # Wall repulsion force
    if ribo_coords is not None:
        print(f"  Adding wall repulsion force ({len(ribo_coords)} ribosome atoms)...")
        ribo_tree = KDTree(ribo_coords)
        n_atoms = modeller.topology.getNumAtoms()
        positions = modeller.positions

        mrna_coords_nm = np.array([positions[i].value_in_unit(nanometer)
                                    for i in range(n_atoms)])
        mrna_coords_A = mrna_coords_nm * 10.0

        _, nearest_idx = ribo_tree.query(mrna_coords_A)
        nearest_ribo_nm = ribo_coords[nearest_idx] * 0.1

        wall_force = CustomExternalForce(
            "0.5*k_wall*step(r_min-dist)*((r_min-dist)^2);"
            "dist=sqrt((x-wx)^2+(y-wy)^2+(z-wz)^2);"
            "r_min=0.3"
        )
        wall_force.addGlobalParameter("k_wall", 1000.0)
        wall_force.addPerParticleParameter("wx")
        wall_force.addPerParticleParameter("wy")
        wall_force.addPerParticleParameter("wz")

        for i in range(n_atoms):
            wall_force.addParticle(i, [
                nearest_ribo_nm[i, 0], nearest_ribo_nm[i, 1], nearest_ribo_nm[i, 2]
            ])
        system.addForce(wall_force)

    # Select best available platform (CUDA > OpenCL > CPU)
    for platform_name in ['CUDA', 'OpenCL', 'CPU']:
        try:
            platform = Platform.getPlatformByName(platform_name)
            print(f"  Using {platform_name} platform")
            break
        except Exception:
            continue

    integrator = LangevinMiddleIntegrator(
        PHASE1_TEMP * kelvin, 1 / picosecond, 0.002 * picoseconds)
    sim = Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Minimize
    state0 = sim.context.getState(getEnergy=True)
    print(f"  Initial energy: {state0.getPotentialEnergy()}")
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=1000)
    state1 = sim.context.getState(getEnergy=True)
    print(f"  Post-minimize: {state1.getPotentialEnergy()}")

    # Phase 1: 400K
    print(f"  Phase 1: {PHASE1_STEPS} steps @ {PHASE1_TEMP}K...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(PHASE1_STEPS // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(PHASE1_STEPS)

    # Phase 2: 350K
    print(f"  Phase 2: {PHASE2_STEPS} steps @ {PHASE2_TEMP}K...")
    integrator.setTemperature(PHASE2_TEMP * kelvin)
    sim.step(PHASE2_STEPS)

    # Phase 3: 310K
    print(f"  Phase 3: {PHASE3_STEPS} steps @ {PHASE3_TEMP}K...")
    integrator.setTemperature(PHASE3_TEMP * kelvin)
    sim.step(PHASE3_STEPS)

    # Quench
    print("  Final minimization (quench)...")
    sim.minimizeEnergy(maxIterations=1000)
    state_final = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"  Final energy: {state_final.getPotentialEnergy()}")

    with open(output_pdb, "w") as f:
        OmmPDB.writeFile(sim.topology, state_final.getPositions(), f, keepIds=True)
    print(f"  Written: {output_pdb}")


@app.function(gpu="T4", timeout=600)
def relax_on_gpu(raw_pdb_content: str) -> str:
    """Run 500K-step MD annealing on GPU with ribosome wall repulsion.

    Fetches 6Y0G from RCSB for ribosome context (self-contained).
    """
    import numpy as np
    from scipy.spatial import KDTree
    from biotite.structure import AtomArrayStack
    from biotite.structure.io.pdb import PDBFile
    import biotite.database.rcsb as rcsb_db
    import biotite.structure.io.pdbx as pdbx

    input_path = "/tmp/raw_mrna.pdb"
    output_path = "/tmp/relaxed_mrna.pdb"

    with open(input_path, "w") as f:
        f.write(raw_pdb_content)

    # Fetch 6Y0G for ribosome context
    print("=== Fetching 6Y0G for ribosome context ===")
    cif_path = rcsb_db.fetch("6Y0G", "cif", target_path="/tmp")
    cif = pdbx.CIFFile.read(cif_path)
    full_arr = pdbx.get_structure(cif, model=1)
    if isinstance(full_arr, AtomArrayStack):
        full_arr = full_arr[0]

    # Read mRNA coords from raw PDB
    mrna_pdb = PDBFile.read(input_path)
    mrna_arr = mrna_pdb.get_structure()
    if isinstance(mrna_arr, AtomArrayStack):
        mrna_arr = mrna_arr[0]
    mrna_coords = mrna_arr.coord

    # Extract nearby ribosome atoms
    mask_ribo = np.isin(full_arr.chain_id, RIBOSOME_CHAINS)
    ribo_coords = full_arr[mask_ribo].coord
    print(f"  Ribosome: {len(ribo_coords)} atoms total")

    mrna_tree = KDTree(mrna_coords)
    dists, _ = mrna_tree.query(ribo_coords)
    nearby_ribo = ribo_coords[dists < 15.0]
    print(f"  Nearby ribosome atoms (within 15A): {len(nearby_ribo)}")

    # Run MD relaxation
    _relax_rna(input_path, output_path, ribo_coords=nearby_ribo)

    with open(output_path) as f:
        return f.read()


@app.local_entrypoint()
def main():
    import os

    input_pdb = "extended_mrna.pdb"
    if not os.path.exists(input_pdb):
        print(f"ERROR: {input_pdb} not found.")
        print("Run: python3.11 build_extended_mrna.py --skip-minimize")
        return

    with open(input_pdb) as f:
        raw_content = f.read()

    print(f"Sending {input_pdb} to Modal GPU for relaxation...")
    relaxed_content = relax_on_gpu.remote(raw_content)

    with open(input_pdb, "w") as f:
        f.write(relaxed_content)
    print(f"Done! Relaxed structure written to {input_pdb}")
