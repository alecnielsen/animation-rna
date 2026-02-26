"""GPU-accelerated mRNA relaxation and rendering via Modal.

GPU functions:
  relax_on_gpu    — MD annealing on T4 (configurable steps/temps)
  render_on_gpu   — Blender Cycles render from pre-baked .blend file
  animate_on_gpu  — Render frame range from pre-baked .blend file

Usage:
  # Full pipeline: relax + render (debug)
  modal run modal_gpu.py

  # Render only (skip relaxation, use existing scene.blend)
  modal run modal_gpu.py --skip-relax

  # Relax only (no render)
  modal run modal_gpu.py --skip-render

  # Production render (1920x1080, 128 samples)
  modal run modal_gpu.py --skip-relax --production

  # Render animation frames (requires scene.blend with animation setup)
  modal run modal_gpu.py --skip-relax --animate --frame-start 0 --frame-end 239

Prerequisites:
  pip install modal
  modal setup  # one-time auth
  python3.11 render_single_frame.py --save-blend  # create scene.blend
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


def _relax_rna(input_pdb, output_pdb, ribo_coords=None,
               phase1_steps=3000000, phase1_temp=500,
               phase2_steps=1000000, phase2_temp=400,
               phase3_steps=1000000, phase3_temp=310):
    """3-stage annealing with wall repulsion — GPU-accelerated.

    Default parameters are aggressive (5M total steps, 500K peak) to
    break tile periodicity in the extended mRNA. For quick tests, pass
    lower values (e.g., phase1_steps=300000, phase1_temp=400).
    """
    import sys
    import numpy as np
    from scipy.spatial import KDTree
    from openmm.app import (
        PDBFile as OmmPDB, Simulation, CutoffNonPeriodic, HBonds,
        StateDataReporter,
    )
    from openmm import LangevinMiddleIntegrator, CustomExternalForce, Platform
    from openmm.unit import kelvin, picosecond, picoseconds, nanometer

    PHASE1_STEPS = phase1_steps
    PHASE1_TEMP = phase1_temp
    PHASE2_STEPS = phase2_steps
    PHASE2_TEMP = phase2_temp
    PHASE3_STEPS = phase3_steps
    PHASE3_TEMP = phase3_temp
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


@app.function(gpu="T4", image=md_image, timeout=1800)
def relax_on_gpu(
    raw_pdb_content: str,
    phase1_steps: int = 3000000,
    phase1_temp: int = 500,
    phase2_steps: int = 1000000,
    phase2_temp: int = 400,
    phase3_steps: int = 1000000,
    phase3_temp: int = 310,
) -> str:
    """Run MD annealing on GPU with ribosome wall repulsion.

    Default: 5M steps (3M@500K + 1M@400K + 1M@310K) for aggressive
    symmetry breaking. Pass lower values for quick tests.
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

    _relax_rna(input_path, output_path, ribo_coords=nearby_ribo,
               phase1_steps=phase1_steps, phase1_temp=phase1_temp,
               phase2_steps=phase2_steps, phase2_temp=phase2_temp,
               phase3_steps=phase3_steps, phase3_temp=phase3_temp)

    with open(output_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Blender Rendering
# ---------------------------------------------------------------------------

@app.function(gpu="T4", image=render_image, timeout=600)
def render_on_gpu(
    blend_data: bytes,
    debug: bool = True,
) -> bytes:
    """Render a single frame from a pre-baked .blend file on GPU.

    The .blend file contains the fully set-up scene (molecules, styles,
    materials, camera). We just open it, configure CUDA, and render.
    This takes ~1-2 minutes vs 10+ minutes for full scene setup.
    """
    import os

    workdir = "/tmp/render"
    os.makedirs(f"{workdir}/renders", exist_ok=True)

    blend_path = f"{workdir}/scene.blend"
    with open(blend_path, "wb") as f:
        f.write(blend_data)
    print(f"=== Loaded .blend file ({len(blend_data) / 1024 / 1024:.1f} MB) ===")

    import bpy

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    # Configure CUDA GPU rendering
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = True
    scene.cycles.device = 'GPU'
    print(f"  GPU rendering enabled (CUDA)")

    # Override samples for debug mode
    if debug:
        scene.cycles.samples = 32
        scene.render.resolution_x = 960
        scene.render.resolution_y = 540

    # Enable denoising for cleaner output
    scene.cycles.use_denoising = True

    output_path = f"{workdir}/renders/single_frame.png"
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'PNG'

    print(f"  Rendering ({scene.render.resolution_x}x{scene.render.resolution_y}, "
          f"{scene.cycles.samples} samples)...")
    bpy.ops.render.render(write_still=True)
    print(f"  Render complete: {output_path}")

    with open(output_path, "rb") as f:
        return f.read()


@app.function(gpu="T4", image=render_image, timeout=3600)
def animate_on_gpu(
    blend_data: bytes,
    frame_start: int = 0,
    frame_end: int = 239,
) -> list[bytes]:
    """Render a range of animation frames from a pre-baked .blend file on GPU.

    Returns a list of PNG bytes, one per frame.
    """
    import os

    workdir = "/tmp/render"
    os.makedirs(f"{workdir}/frames", exist_ok=True)

    blend_path = f"{workdir}/scene.blend"
    with open(blend_path, "wb") as f:
        f.write(blend_data)
    print(f"=== Loaded .blend file ({len(blend_data) / 1024 / 1024:.1f} MB) ===")

    import bpy

    bpy.ops.wm.open_mainfile(filepath=blend_path)
    scene = bpy.context.scene

    # Configure CUDA GPU rendering
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'CUDA'
    cycles_prefs.get_devices()
    for device in cycles_prefs.devices:
        device.use = True
    scene.cycles.device = 'GPU'
    scene.cycles.use_denoising = True
    print(f"  GPU rendering enabled (CUDA)")

    scene.render.image_settings.file_format = 'PNG'
    frames_data = []

    for frame_num in range(frame_start, frame_end + 1):
        scene.frame_set(frame_num)
        frame_path = f"{workdir}/frames/frame_{frame_num:04d}.png"
        scene.render.filepath = frame_path
        print(f"  Rendering frame {frame_num}/{frame_end}...")
        bpy.ops.render.render(write_still=True)
        with open(frame_path, "rb") as f:
            frames_data.append(f.read())

    print(f"  Rendered {len(frames_data)} frames")
    return frames_data


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    skip_relax: bool = False,
    skip_render: bool = False,
    production: bool = False,
    animate: bool = False,
    frame_start: int = 0,
    frame_end: int = 239,
):
    import os

    mrna_pdb_path = "extended_mrna.pdb"
    blend_path = "scene.blend"
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
        if not os.path.exists(blend_path):
            print(f"ERROR: {blend_path} not found.")
            print("Run: python3.11 render_single_frame.py --save-blend")
            return

        with open(blend_path, "rb") as f:
            blend_data = f.read()
        print(f"Loaded {blend_path} ({len(blend_data) / 1024 / 1024:.1f} MB)")

        if animate:
            print(f"=== Rendering animation frames {frame_start}-{frame_end} on GPU... ===")
            frames_data = animate_on_gpu.remote(blend_data, frame_start, frame_end)

            os.makedirs("renders/frames", exist_ok=True)
            for i, png_data in enumerate(frames_data):
                frame_num = frame_start + i
                frame_path = f"renders/frames/frame_{frame_num:04d}.png"
                with open(frame_path, "wb") as f:
                    f.write(png_data)
            print(f"Saved {len(frames_data)} frames to renders/frames/")
        else:
            debug = not production
            mode = "production (1920x1080)" if production else "debug (960x540)"
            print(f"=== Rendering single frame on GPU ({mode})... ===")
            png_data = render_on_gpu.remote(blend_data, debug=debug)

            os.makedirs("renders", exist_ok=True)
            with open(output_png, "wb") as f:
                f.write(png_data)
            print(f"Render saved to {output_png}")

    print("=== Done ===")
