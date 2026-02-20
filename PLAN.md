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

Two-pass compositing per frame (see STYLE.md for details):
1. Render internal atoms (cartoon) — mRNA, tRNAs, polypeptide
2. Render ribosome surface (flat opaque)
3. Composite: translucent surface overlay + edge outline

For animation, each frame is rendered as two passes and composited.

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

- Build a long alpha helix (~20+ residues) pre-threaded through the exit
  tunnel, extending off-screen on the exit side.
- Each cycle: one new residue is "added" at the ribosome end (progressive
  reveal via clip plane or geometry nodes mask).
- After N cycles, N residues have been added, but the off-screen end absorbs
  the growth invisibly. Visual state matches frame 0.

### tRNA cycling

- tRNAs naturally loop: each cycle ends with P-site occupied, A-site empty.
- Same geometry (chain B4) recycled for every incoming tRNA.

## Animation approach: Two-layer choreography + thermal motion

### Layer 1: Scripted choreography (rigid-body keyframes)

The elongation cycle is scripted as rigid-body transforms. The ribosome
stays static (frame of reference). Moving parts:

1. **mRNA** — ratchets one codon per cycle along the mRNA principal axis
2. **tRNA (incoming)** — glides from outside → A-site
3. **tRNA (P→E)** — translocates P→E site, then departs
4. **tRNA (A→P)** — translocates A→P site (becomes new P-site tRNA)
5. **Polypeptide** — progressive reveal of one residue per cycle

### Layer 2: ProDy NMA thermal texture

Replace hand-rolled jitter with physically realistic per-atom displacements
from Normal Mode Analysis:

1. Compute ANM modes from 6Y0G with ProDy (`sparse=True`, C-alpha/P atoms)
2. Pre-generate displacement frames via `sampleModes()`
3. Extend to all atoms with `extendModel()`
4. Each render frame: `positions = original + choreography_delta + NMA_displacement`

NMA inherently encodes structural constraints: atoms buried in the ribosome
barely move, exposed mRNA ends and free tRNAs fluctuate more.

### Sequence (one elongation cycle, frames scaled to total)

```
Phase 1: ESTABLISH (0-12%)
         P-site tRNA holds peptide. A-site empty.

Phase 2: tRNA DELIVERY (12-38%)
         Aminoacyl-tRNA glides into A-site from outside.

Phase 3: ACCOMMODATION (38-50%)
         tRNA settles into A-site.

Phase 4: PEPTIDE TRANSFER (50-62%)
         Polypeptide grows by one residue (progressive reveal).

Phase 5: TRANSLOCATION (62-88%)
         A-site tRNA → P-site (with peptide).
         P-site tRNA → E-site (deacylated).
         mRNA advances one codon.

Phase 6: tRNA DEPARTURE (88-100%)
         E-site tRNA departs.
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

### v2 (current)
- [ ] Extended mRNA: procedurally build long strand with biotite
- [ ] Extended polypeptide: long alpha helix for progressive reveal
- [ ] 10-cycle choreography with seamless loop
- [ ] ProDy NMA thermal motion (replace hand-rolled jitter)
- [ ] Camera orbit fix (centroid pivot)
- [ ] Full production render

## Tech stack

- **Python 3.11** — required by Blender
- **Molecular Nodes 4.5.10** — PDB loading, molecular styles
- **bpy (Blender Python)** — headless rendering, materials, keyframes
- **Cycles renderer** — lighting, materials
- **PIL / numpy** — per-frame compositing (translucent overlay + edge outline)
- **ffmpeg** — final video encoding
- **ProDy** — Normal Mode Analysis for thermal motion (v2)
- **biotite** — procedural mRNA/polypeptide construction (v2)
