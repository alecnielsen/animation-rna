"""GPU-accelerated mRNA relaxation and rendering via Modal.

Two GPU functions:
  relax_on_gpu  — 500K-step MD annealing on T4 (~2 min vs ~50 min CPU)
  render_on_gpu — Blender Cycles render on T4 with CUDA

Usage:
  # Full pipeline: relax + render (debug)
  modal run modal_gpu.py

  # Render only (skip relaxation, use existing extended_mrna.pdb)
  modal run modal_gpu.py --skip-relax

  # Relax only (no render)
  modal run modal_gpu.py --skip-render

  # Production render (1920x1080, 128 samples)
  modal run modal_gpu.py --skip-relax --production

Prerequisites:
  pip install modal
  modal setup  # one-time auth
  python3.11 build_extended_mrna.py --skip-minimize  # raw tiled mRNA
"""

import modal

# --- Images ---

md_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install("openmm", "cudatoolkit", channels=["conda-forge"])
    .pip_install("biotite", "scipy", "numpy")
)

render_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1-mesa-glx", "libxi6", "libxrender1", "libxfixes3",
        "libxkbcommon0", "libsm6", "libice6", "libgomp1", "libxxf86vm1",
        "libxext6", "libx11-6", "libgl1",
    )
    .pip_install("molecularnodes[bpy]", "numpy")
)

app = modal.App("animation-rna")

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


# ---------------------------------------------------------------------------
# MD Relaxation (same as modal_relax.py)
# ---------------------------------------------------------------------------

def _load_and_prepare(input_pdb):
    """Load RNA PDB into OpenMM, fix termini, add hydrogens."""
    import tempfile
    import os
    from openmm.app import PDBFile as OmmPDB, ForceField, Modeller

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

    state0 = sim.context.getState(getEnergy=True)
    print(f"  Initial energy: {state0.getPotentialEnergy()}")
    print("  Minimizing...")
    sim.minimizeEnergy(maxIterations=1000)
    state1 = sim.context.getState(getEnergy=True)
    print(f"  Post-minimize: {state1.getPotentialEnergy()}")

    print(f"  Phase 1: {PHASE1_STEPS} steps @ {PHASE1_TEMP}K...")
    sim.reporters.append(
        StateDataReporter(sys.stdout, max(PHASE1_STEPS // 5, 1), step=True,
                          potentialEnergy=True, temperature=True, speed=True)
    )
    sim.step(PHASE1_STEPS)

    print(f"  Phase 2: {PHASE2_STEPS} steps @ {PHASE2_TEMP}K...")
    integrator.setTemperature(PHASE2_TEMP * kelvin)
    sim.step(PHASE2_STEPS)

    print(f"  Phase 3: {PHASE3_STEPS} steps @ {PHASE3_TEMP}K...")
    integrator.setTemperature(PHASE3_TEMP * kelvin)
    sim.step(PHASE3_STEPS)

    print("  Final minimization (quench)...")
    sim.minimizeEnergy(maxIterations=1000)
    state_final = sim.context.getState(getEnergy=True, getPositions=True)
    print(f"  Final energy: {state_final.getPotentialEnergy()}")

    with open(output_pdb, "w") as f:
        OmmPDB.writeFile(sim.topology, state_final.getPositions(), f, keepIds=True)
    print(f"  Written: {output_pdb}")


@app.function(gpu="T4", image=md_image, timeout=600)
def relax_on_gpu(raw_pdb_content: str) -> str:
    """Run 500K-step MD annealing on GPU with ribosome wall repulsion."""
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

    print("=== Fetching 6Y0G for ribosome context ===")
    cif_path = rcsb_db.fetch("6Y0G", "cif", target_path="/tmp")
    cif = pdbx.CIFFile.read(cif_path)
    full_arr = pdbx.get_structure(cif, model=1)
    if isinstance(full_arr, AtomArrayStack):
        full_arr = full_arr[0]

    mrna_pdb = PDBFile.read(input_path)
    mrna_arr = mrna_pdb.get_structure()
    if isinstance(mrna_arr, AtomArrayStack):
        mrna_arr = mrna_arr[0]
    mrna_coords = mrna_arr.coord

    mask_ribo = np.isin(full_arr.chain_id, RIBOSOME_CHAINS)
    ribo_coords = full_arr[mask_ribo].coord
    print(f"  Ribosome: {len(ribo_coords)} atoms total")

    mrna_tree = KDTree(mrna_coords)
    dists, _ = mrna_tree.query(ribo_coords)
    nearby_ribo = ribo_coords[dists < 15.0]
    print(f"  Nearby ribosome atoms (within 15A): {len(nearby_ribo)}")

    _relax_rna(input_path, output_path, ribo_coords=nearby_ribo)

    with open(output_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Blender Rendering
# ---------------------------------------------------------------------------

@app.function(gpu="T4", image=render_image, timeout=1800)
def render_on_gpu(
    render_script: str,
    mrna_pdb: str,
    peptide_pdb: str,
    debug: bool = True,
) -> bytes:
    """Run render_single_frame.py on GPU via Blender Cycles CUDA."""
    import subprocess
    import os

    workdir = "/app"
    os.makedirs(f"{workdir}/renders", exist_ok=True)

    with open(f"{workdir}/extended_mrna.pdb", "w") as f:
        f.write(mrna_pdb)
    with open(f"{workdir}/tunnel_polypeptide.pdb", "w") as f:
        f.write(peptide_pdb)
    with open(f"{workdir}/render_single_frame.py", "w") as f:
        f.write(render_script)

    args = ["python3", f"{workdir}/render_single_frame.py", "--gpu"]
    if debug:
        args.append("--debug")

    print(f"=== Running: {' '.join(args)} ===")
    result = subprocess.run(args, cwd=workdir, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")
    if result.returncode != 0:
        raise RuntimeError(f"Render failed (exit {result.returncode})")

    output_path = f"{workdir}/renders/single_frame.png"
    with open(output_path, "rb") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    skip_relax: bool = False,
    skip_render: bool = False,
    production: bool = False,
):
    import os

    mrna_pdb_path = "extended_mrna.pdb"
    peptide_pdb_path = "tunnel_polypeptide.pdb"
    output_png = "renders/single_frame.png"

    # --- Relax ---
    if not skip_relax:
        if not os.path.exists(mrna_pdb_path):
            print(f"ERROR: {mrna_pdb_path} not found.")
            print("Run: python3.11 build_extended_mrna.py --skip-minimize")
            return

        with open(mrna_pdb_path) as f:
            raw_content = f.read()

        print(f"=== Relaxing mRNA on GPU... ===")
        relaxed_content = relax_on_gpu.remote(raw_content)

        with open(mrna_pdb_path, "w") as f:
            f.write(relaxed_content)
        print(f"Relaxed structure written to {mrna_pdb_path}")

    # --- Render ---
    if not skip_render:
        for path in [mrna_pdb_path, peptide_pdb_path]:
            if not os.path.exists(path):
                print(f"ERROR: {path} not found.")
                return

        with open("render_single_frame.py") as f:
            render_script = f.read()
        with open(mrna_pdb_path) as f:
            mrna_content = f.read()
        with open(peptide_pdb_path) as f:
            peptide_content = f.read()

        debug = not production
        mode = "production (1920x1080)" if production else "debug (960x540)"
        print(f"=== Rendering single frame on GPU ({mode})... ===")
        png_data = render_on_gpu.remote(
            render_script, mrna_content, peptide_content, debug=debug,
        )

        os.makedirs("renders", exist_ok=True)
        with open(output_png, "wb") as f:
            f.write(png_data)
        print(f"Render saved to {output_png}")

    print("=== Done ===")
