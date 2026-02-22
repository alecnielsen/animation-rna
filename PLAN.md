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
- Ribosome: translucent via backface-culling shader (~35% opacity)
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

### Layer 2: Structural deformation + thermal motion

- **PCA modes:** Pre-computed from MD trajectories, applied as per-residue
  displacement via integer-harmonic sines. Gives physically realistic
  backbone undulation that loops perfectly. Amplitude 1.5 BU (30:1 ratio
  over residue jitter).
- **Per-residue jitter:** Coherent sum-of-sines displacement grouped by
  residue (all atoms in a residue move together). Surface meshes re-evaluate
  cleanly.
- **Ribosome jitter:** Visible rigid-body translation (0.15 BU) + rotation (5deg).
- **tRNA tumbling:** Full rotational freedom during approach/departure,
  smooth decay during accommodation, 5% residual tumble when bound.

For seamless looping, all frequencies are integer harmonics of the total
animation period (k/T) so every component returns to its exact starting
phase at the loop point.

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

### v4 (current)
- [x] All molecules use StyleSurface (unified realistic look)
- [x] Single-pass rendering with shader transparency (proper depth occlusion)
- [x] Per-residue deformation (replaces broken per-atom jitter on surface meshes)
- [x] Increase ribosome jitter 5x (0.15 BU trans, 5deg rot)
- [x] Increase PCA amplitude 3x (1.5 BU base, 30:1 ratio over jitter)
- [x] Extended tRNA tumbling windows + 5% residual tumble when bound
- [x] Enhanced mRNA MD: 500K steps, 3-stage anneal, sequence randomization
- [x] Eliminate composite.py from pipeline (single-pass handles occlusion)
- [ ] Debug render validation
- [ ] Full production render

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
