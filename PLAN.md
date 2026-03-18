# Animation Plan

## Goal

Seamlessly looping video of protein translation (~10 elongation cycles) for a product webpage.
Human 80S ribosome (PDB 6Y0G) with extended mRNA, tRNAs cycling, and visible polypeptide growth.

## PDB structure

### 6Y0G — Human 80S ribosome, classical PRE state (3.2 A)

| Chain(s) | Component |
|----------|-----------|
| `S*` (34 chains) | 40S small subunit (18S rRNA + proteins) |
| `L*` (45 chains) | 60S large subunit (28S/5.8S/5S rRNA + proteins) |
| `A4` | mRNA (poly-U, 28 nt) |
| `B4` | P-site tRNA |
| `D4` | A-site tRNA |
| `C4` | Nascent peptide (dipeptide) |

## Rendering approach

Single-pass Cycles rendering with shader-based transparency (see STYLE.md):
- All molecules render simultaneously with correct depth occlusion
- Ribosome: translucent via Principled BSDF Alpha=0.06 (HASHED blend mode)
- Internal molecules: opaque StyleSurface with emission
- No compositing step needed

## Seamless loop design

The animation shows ~10 elongation cycles and loops perfectly. This works
via the "conveyor belt" principle: both mRNA and polypeptide extend far
beyond the camera frame on both sides. After N cycles of ratcheting, the
visual state is identical to frame 0 because the N-unit shift is entirely
absorbed off-screen.

### mRNA conveyor

- Build one long continuous mRNA strand (~50+ codons) with realistic
  backbone geometry (varying sequence for natural kinks, not just poly-U).
- Position so both ends extend well beyond the camera frame.
- Ratchet by one codon per cycle (N codons total over the animation).
- After N codons of shift, the visible portion looks identical to the start
  because the cartoon representation is visually uniform.

### Polypeptide conveyor

- Build a long alpha helix (~80+ residues) pre-threaded through the exit
  tunnel, extending off-screen on the exit side.
- Each cycle: one new residue is "added" at the ribosome end (progressive
  reveal via geometry mask).
- After N cycles, N residues have been added, but the off-screen end absorbs
  the growth invisibly. Visual state matches frame 0.

### tRNA cycling

- tRNAs naturally loop: each cycle ends with P-site occupied, A-site empty.
- Same geometry (chain B4) recycled for every incoming tRNA.

## Animation approach: Two-layer choreography + thermal motion

### Layer 1: Scripted choreography (rigid-body keyframes)

The elongation cycle is scripted as rigid-body transforms. The ribosome
stays near-static (subtle jitter only). Moving parts:

1. **mRNA** — ratchets one codon per cycle (translation only, no rotation)
2. **tRNA (incoming)** — glides from outside → A-site, tumbles in solution
3. **tRNA (P→E)** — translocates P→E site, then departs with tumbling
4. **tRNA (A→P)** — translocates A→P site (becomes new P-site tRNA)
5. **Polypeptide** — progressive reveal of one residue per cycle (no rigid-body motion)

### Layer 2: Per-frame OpenMM thermal motion

- **Elastic network MD:** Per-frame OpenMM simulation with harmonic restraints
  (k=100 kJ/mol/nm²) for mRNA and tRNAs. Produces physically realistic
  thermal breathing without tearing surface meshes.
- **Ribosome ENM trajectory:** Pre-computed 456-frame elastic network mode
  trajectory (16938 residues) loaded from `ribosome_thermal.npz`.
  Computed on Modal GPU for performance.
- **tRNA tumbling:** Full rotational freedom during approach/departure,
  smooth decay during accommodation, 5% residual tumble when bound.

### Sequence (one elongation cycle, frames scaled to total)

```
Phase 1: ESTABLISH (0-5%, f0-f12)
         P-site tRNA holds peptide. A-site empty.

Phase 2: tRNA DELIVERY (5-40%, f12-f96)
         Aminoacyl-tRNA glides into A-site from outside (tumbling).

Phase 3: ACCOMMODATION (40-50%, f96-f120)
         tRNA settles into A-site (tumble decays to residual 5%).

Phase 4: PEPTIDE TRANSFER (50-60%, f120-f144)
         Polypeptide grows by one residue (progressive reveal).

Phase 5: TRANSLOCATION (60-80%, f144-f192)
         A-site tRNA → P-site (with peptide).
         P-site tRNA → E-site (deacylated).
         mRNA advances one codon.

Phase 6: tRNA DEPARTURE (80-100%, f192-f240)
         E-site tRNA departs (tumble ramps up from residual).
         State is identical to Phase 1 → next cycle begins.
```

This cycle repeats N times (~10). Total frame count TBD based on pacing.

### Camera

Slow continuous orbit over the full animation (disabled for now, needs
proper centroid-based pivot). Slightly angled to show the exit tunnel.

## Milestones

### v1 (complete)
- [x] Environment setup (Python 3.11 + Molecular Nodes + headless bpy)
- [x] Basic test render of 6Y0G
- [x] Visual style: two-pass compositing, cartoon internals, surface outline
- [x] Full 6Y0G ribosome single-frame render
- [x] Measure chain centroids (`measure_positions.py`)
- [x] Animation script: 1-cycle elongation (`animate.py`)
- [x] Compositing + encoding pipeline (`composite.py`, `encode.py`)
- [x] Debug render test (480x270, 24 frames)

### v2 (complete)
- [x] Extended mRNA: procedurally build long strand with biotite
  - [x] `build_extended_mrna.py`: tiles chain A4 x10 with correct backbone spacing
  - [x] OpenMM MD relaxation at 400K to break tile symmetry
  - [x] `animate.py`: loads extended mRNA from local PDB
- [x] Extended polypeptide: ~30 residue polyalanine alpha helix
  - [x] `build_extended_polypeptide.py`: ideal helix geometry, aligned to C4 position
  - [x] Progressive reveal in animate.py (1 residue per cycle)
- [x] 10-cycle choreography with seamless loop
  - [x] Nested loop: N_CYCLES x FRAMES_PER_CYCLE
  - [x] Cumulative mRNA offset per cycle
  - [x] Loop-safe integer-harmonic jitter frequencies

### v3 (complete)
- [x] Remove edge outline, increase surface opacity to 35%
- [x] Ribosome jitter (subtle rigid-body motion)
- [x] mRNA: remove rigid-body rotation, reduce per-atom jitter
- [x] mRNA: PCA structural modes for backbone undulation
- [x] Extended MD relaxation (200K steps, annealing protocol)
- [x] tRNA tumbling during approach/departure
- [x] Polypeptide: remove rigid-body jitter and choreographic motion
- [x] Tunnel-threaded polypeptide (void-tracing through 60S)
- [x] PCA modes for tRNA structural deformation

### v4 (complete)
- [x] All molecules use StyleSurface (unified realistic look)
- [x] Single-pass rendering with shader transparency (proper depth occlusion)
- [x] Per-residue deformation (replaces broken per-atom jitter on surface meshes)
- [x] Increase ribosome jitter 5x (0.15 BU trans, 5deg rot)
- [x] Increase PCA amplitude 3x (1.5 BU base, 30:1 ratio over jitter)
- [x] Extended tRNA tumbling windows + 5% residual tumble when bound
- [x] Enhanced mRNA MD: 500K steps, 3-stage anneal, sequence randomization
- [x] Eliminate composite.py from pipeline (single-pass handles occlusion)

### v5 (complete)
- [x] Ribosome translucency: switch to Principled BSDF Alpha=0.06 (HASHED blend mode)
- [x] Tested and rejected alternative translucency approaches (mix-shader, Transmission, Fresnel, Glass BSDF)
- [x] Tunnel-threaded polypeptide: C4 path + forward-biased scoring, extended 100 steps (200 A) past exit
- [x] Single-frame renderer (`render_single_frame.py`) for validating molecule placements
- [x] Camera zoom tuned to 85% auto-frame distance

### v6 (complete)
- [x] Repeating HP35 polypeptide domains (8x Villin HP35 with GSG linkers)
- [x] Dual-coordinate NPZ for polypeptide morph animation (extended + folded per domain)
- [x] mRNA decoding center alignment (shift extended mRNA to match tRNA-mRNA base pairing)
- [x] Camera angle from Blender viewport: polypeptide parallel to view plane
- [x] 5M-step GPU relaxation (Modal) for mRNA
- [x] CONECT record handling: strip bad CONECT records so MN infers bonds by template

### v7 (complete)
- [x] Per-frame OpenMM thermal motion (replaces PCA modes — no more mesh tearing)
  - [x] Elastic network restraints (k=100 kJ/mol/nm²) for mRNA, tRNAs
  - [x] Pre-computed ENM thermal trajectory for ribosome (456 frames, 16938 residues)
  - [x] MD parameterization fix: remove OP3 from tRNA 5' terminus
- [x] 2-pass composite rendering: cartoon for ribosome (outline via dilated silhouette), ribbon for tRNA, cartoon for mRNA/peptide backbone
- [x] mRNA stationary (no rigid-body sliding — ratchet absorbed off-screen)
- [x] Å→BU scale fix: world_scale=0.01 (not 0.1), relative polypeptide morph
- [x] Polypeptide folding morph: domain 0 unfolds N→C wave over 38 cycles, domains 1-7 stay folded
- [x] Gradient scroll: anchor tunnel residues at PTC, smooth ramp to full scroll for external residues
- [x] GLY-aware folded residue ranges (4 atoms vs 5 for standard amino acids)
- [x] Double-folding morph inversion: mesh starts folded from PDB, apply unfold displacement (not fold)
- [x] Distal tail extended from 10 to 40 residues
- [x] Resume rendering support (skip already-rendered frames)

### v8 (complete)
- [x] N-to-C wave fold timing: each residue folds within a WAVE_WINDOW=0.5 window, staggered along chain
- [x] Replaced global fold progress with per-cycle local timing (cycle_t = local_frame / frames_per_cycle)

### v9 (complete)
- [x] Emergence-based polypeptide folding: residues emerge from tunnel progressively, fold within 6-cycle window
- [x] Tunnel anchor collapse: unemerged residues collapse to tunnel exit point
- [x] Kept only ~6 residues extended at any time (~23 Å) to reduce overlap with folded domains

### v10 (complete)
- [x] Compact unfold toward tunnel: replace extended-coord morph with directional displacement
- [x] Per-residue displacement along -scroll_vector (toward tunnel), capped at 15 Å (MAX_UNFOLD_BU=0.15)
- [x] chain_frac scaling: N-terminal residues (near tunnel) move most, C-terminal (near folded stack) stays anchored
- [x] Reverted to v8 N-to-C wave timing (removed v9 emergence/cycle logic)
- [x] Eliminated all extended coordinate usage for domain_0 — no more 131 Å extended chain
- [x] Fix: inverted chain_frac so displacement anchors at folded stack (not tunnel)
- [x] Fix: global fold timing (single fold across animation, not per-cycle repetition)

### v11 (complete)
- [x] Polypeptide thermal jitter via OpenMM elastic network MD
  - [x] Build all-atom PDB via PDBFixer (add missing sidechains to backbone-only PDB, fix GLY CB)
  - [x] Initialize MolecularDynamics for polypeptide (same pattern as mRNA/tRNAs)
  - [x] MD deltas layered on top of morph+scroll positions (existing wiring at line ~1203)
  - [x] Loop-blend for seamless looping (inherited from MolecularDynamics class)
- [x] Swap polyalanine to HP35 sequence throughout entire chain
  - [x] Tunnel region, tail, extended segments all use cycling HP35 sequence
  - [x] GLY residues correctly have 4 atoms (no CB)
  - [x] Rebuilt all-atom PDB with proper diverse sidechains (6158 atoms)
- Note: peptide MD (6158 atoms) causes occasional slow frames (~300-900s vs ~30s normal),
  likely from OpenMM resolving bad sidechain states. Render still completes, ~5h total.

### v12 (complete)
- [x] mRNA: HP35 coding sequence (replace poly-U)
  - [x] Generate repeating HP35 codons (A=75, U=51, G=111, C=103 across 340 nt)
  - [x] Strip base atoms from tiled backbone, reconstruct via PDBFixer
  - [x] OpenMM parameterization verified (10971 atoms with H)
- [x] mRNA: backbone straightening via P-P-P angle restraints during relaxation
  - [x] Added CustomAngleForce (k=50 kJ/mol/rad², theta0=pi) on 337 P-P-P triplets
  - [x] Local 500K-step relaxation complete (mean angle: 63.9° → 16.1°, max: 131.8° → 59.5°)
- [x] Load full mRNA atoms for surface mesh (skip backbone stripping when style=surface)
- [x] Surface mesh test render (--style=surface, full atoms) — 6.6 min, looks good
- [x] `--no-jitter` CLI flag for animate.py (clean toggle, no code edits needed)
- [x] Confirmed choreographic motion (tRNA swapping, polypeptide fold) works with and without jitter
- Note: when re-rendering, MUST delete old frames first (resume support skips existing frames)

### v13 (complete)
- [x] Surface mesh for all internal molecules in animate.py (`--style=surface`)
  - [x] mRNA: full atoms (not backbone-stripped) when style=surface
  - [x] tRNA: surface style (replaces ribbon)
  - [x] Polypeptide: surface style (replaces spheres)
  - [x] Ribosome: still cartoon (2-pass outline composite)

### v14 (complete)
- [x] mRNA codon ratchet: physical one-codon-per-cycle advance through ribosome
  - [x] Per-frame codon shift computed from P-atom spacing along backbone PCA axis
  - [x] Phase 5 translocation drives mRNA advance (mRNA + P-site tRNA move together)

### v15 (complete)
- [x] Smooth mRNA ratchet: continuous linear advance (replaces jumpy phase-locked burst)
- [x] Extended mRNA: 30 tiles, asymmetric (20 trailing + 10 leading), ends off-camera
- [x] CENTER_INDEX=20 for PDB coordinate limit compliance

### v16 (complete)
- [x] Keyframe interpolation for mRNA ratchet
  - [x] `relax_mrna_keyframes.py`: generate MD-relaxed keyframes at codon shift positions
  - [x] 9 keyframes (shifts 0,5,10,...,35,38), 100K MD steps each
  - [x] `animate.py`: load keyframes from NPZ, interpolate per-residue centroids
  - [x] Fallback to shift+bend when keyframes not available

### v17 (complete)
- [x] Adaptive wall anchors: multi-round MD with periodic ribosome wall re-query
  - [x] `_update_wall_anchors()` recomputes nearest ribosome atoms every 5K steps via KDTree on ALL ~210K ribosome atoms
  - [x] Minimization + velocity reset after each wall anchor update to prevent NaN
  - [x] Softer wall force (k_wall=1000) with 0.3 nm force cap for stability
  - [x] Applied to both `build_extended_mrna.py` and `relax_mrna_keyframes.py`
- [x] Rebuilt `extended_mrna.pdb` (500K steps, 27 atoms <2.5Å vs 204 pre-relaxation)
- [x] Regenerated `mrna_keyframes.npz` (9 keyframes, 100K steps each, ~91 min total)
- [x] Coordinated whole-structure "breathing" motion via ENM thermal trajectory
  - [x] `compute_mrna_thermal.py`: coarse-grained ENM (one bead per residue, P-atom positions)
  - [x] Softer springs (ENM_K=5, k_restraint=20) for visible collective motion
  - [x] 456-frame trajectory with seamless loop cross-fade
  - [x] Gaussian smoothing (sigma=3 residues) along chain prevents cartoon mesh gaps
  - [x] `animate.py`: loads `mrna_thermal.npz`, applies ENM deltas per-frame (replaces per-atom MD jitter)

### v18 (complete)
- [x] Surface mesh for ALL internal molecules (unified production look)
  - [x] mRNA: segmented into 3 × ~170 residues (~5.5K atoms each) to avoid MN surface mesh segfault on 16K+ atoms
  - [x] tRNA: surface style (replaces ribbon)
  - [x] Polypeptide: surface style (replaces spheres)
  - [x] Ribosome: cartoon (2-pass silhouette outline)
- [x] `_split_mrna_pdb()`: splits extended mRNA PDB into N segments at load time
- [x] Per-frame vertex distribution: compute full mRNA positions, then slice to segment meshes
- [x] 456-frame production render (1920×1080, 128 samples) complete

### v20 (complete)
- [x] Curved tRNA entry/exit paths with per-cycle variety
  - [x] Catmull-Rom spline paths (4 waypoints each) replace straight-line lerps
  - [x] Entry: tRNA swoops in from above + laterally offset toward A-site
  - [x] Exit: tRNA lifts away from E-site in opposite lateral direction (visually distinct from entry)
  - [x] Per-cycle randomized path jitter via `get_cycle_paths(cycle)` — seeded RNG perturbs intermediate waypoints (±1 BU lateral, ±0.6 BU vertical, ±0.5 BU along-axis) while keeping A-site/E-site endpoints fixed
  - [x] Closer waypoints (~2× PA_VEC instead of ~4×) so most of the tRNA path is visible on-camera
  - [x] `smoothstep()` easing on all motion phases (delivery, translocation, departure, mRNA shift)
  - [x] `--frames=N` CLI override for testing temporal resolution independently of debug/prod mode

### v19 (complete)
- [x] Per-frame tRNA-mRNA declash to prevent visual clipping at the decoding center
  - [x] `declash_mrna_from_trna()`: KDTree-based soft repulsion pushes mRNA vertices away from tRNA atoms
  - [x] Transforms to world space (Rz(π/2) rotation + tRNA location offsets), declashes, transforms back
  - [x] Smoothstep falloff: hard push at 6 Å, smooth fade to zero at 10 Å — no pop artifacts
  - [x] Render loop restructured: tRNA positions computed before mRNA distribution for declash input
  - [x] Works across all 38 cycles as mRNA ratchets (consistent ~160-200 vertices pushed per frame)

## Tech stack

- **Python 3.11** — required by Blender
- **Molecular Nodes 4.5.10** — PDB loading, molecular styles
- **bpy (Blender Python)** — headless rendering, materials, keyframes
- **Cycles renderer** — lighting, materials
- **PIL / numpy** — image processing (legacy compositing removed in v4)
- **ffmpeg** — final video encoding
- **OpenMM** — MD simulation for mRNA relaxation, polypeptide relaxation, PCA trajectory generation
- **biotite** — procedural mRNA/polypeptide construction
- **scipy** — cubic spline interpolation for tunnel centerline
