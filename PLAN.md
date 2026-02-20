# Animation Plan

## Goal

~10-15 second looping video of protein translation for a product webpage.
Human 80S ribosome (PDB 6Y0G) with mRNA, tRNAs, and growing polypeptide chain.

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
1. Render internal atoms (ball-and-stick) — mRNA, tRNAs, polypeptide
2. Render ribosome surface (flat opaque)
3. Composite: translucent surface overlay + edge outline

For animation, each frame is rendered as two passes and composited.
This can be parallelized (render all pass-1 frames, then all pass-2 frames).

## Animation approach: Rigid-body choreography

Load 6Y0G and separate into component objects. The ribosome stays static
(it's the frame of reference). Moving parts are animated as rigid bodies
along scripted keyframe paths.

### Moving parts

1. **mRNA** — slides through the ribosome channel. Translate along the mRNA
   axis by ~3 nucleotide widths per cycle (one codon). The mRNA in 6Y0G is
   only 28 nt, so we may need to extend it procedurally (duplicate + offset)
   to avoid running out of strand.

2. **tRNAs** — three copies cycling through positions:
   - **Incoming tRNA**: starts outside the ribosome, glides into A-site
   - **A→P transition**: after peptide transfer, A-site tRNA moves to P-site
   - **P→E transition**: P-site tRNA moves to E-site and departs
   - We extract the tRNA geometry from chains B4/D4 and place copies at
     each position. Keyframe their location/rotation to interpolate between
     sites.

3. **Polypeptide chain** — grows by one amino acid per cycle, emerging from
   the exit tunnel. Approaches:
   - **(a) Progressive reveal:** Pre-build a long polypeptide (e.g. poly-Ala
     helix, ~20 residues). Use a geometry nodes mask or clip plane to reveal
     one residue at a time.
   - **(b) Procedural growth:** Use biotite/MDAnalysis to generate amino acid
     coordinates. Add residues to the Blender mesh at the peptide transfer
     keyframe.
   - **(c) Simple tube:** Animated curve/tube that extends from the exit
     tunnel. Less accurate but visually clean.
   - **Recommended: (a)** — pre-build and progressively reveal. Simplest to
     keyframe and looks good.

### Sequence (one elongation cycle, 240 frames @ 24fps = 10 seconds)

```
Frame 0-30:    ESTABLISH
               Ribosome with mRNA threaded through.
               P-site tRNA holds growing peptide.
               A-site is empty. E-site is empty.

Frame 30-90:   tRNA DELIVERY
               Aminoacyl-tRNA glides into the A-site from outside.
               Enters through the intersubunit space.

Frame 90-120:  ACCOMMODATION
               tRNA settles into A-site.

Frame 120-150: PEPTIDE TRANSFER
               Peptide chain extends by one residue.
               Visual: new amino acid appears on A-site tRNA,
               polypeptide in exit tunnel grows by one unit.

Frame 150-210: TRANSLOCATION
               A-site tRNA → P-site (with peptide)
               P-site tRNA → E-site (deacylated)
               mRNA advances by one codon (~3 nt translation)

Frame 210-240: tRNA DEPARTURE + LOOP RESET
               E-site tRNA departs.
               State is now identical to frame 0 but with
               polypeptide one residue longer.
               Camera position loops seamlessly.
```

### Camera

Slow continuous orbit (~30 degrees over the full 240 frames).
Slightly angled to show the exit tunnel where the polypeptide emerges.
The orbit should loop seamlessly (end angle = start angle + N*360 for some N,
or use a subtle back-and-forth).

### Making it loop

The elongation cycle is inherently repeatable. At frame 240, the state is
the same as frame 0 except:
- The polypeptide is one residue longer (progressive reveal continues)
- The mRNA has advanced one codon

For a seamless loop, either:
- Accept the polypeptide growth as a slow drift (viewers won't notice in 10s)
- Or reset the polypeptide clip plane at the loop point

## Milestones

- [x] Environment setup (Python 3.11 + Molecular Nodes + headless bpy)
- [x] Basic test render of 6Y0G (proof of pipeline)
- [x] Visual style proven on 2PTC test structure
  - [x] Two-pass compositing: translucent surface + edge outline
  - [x] Cartoon for internal components (chosen over ribbon, ball-and-stick, surface)
  - [x] Camera alignment between passes
- [x] Apply style to full 6Y0G ribosome (single frame)
  - [x] Alpha-based silhouette detection (transparent film render)
  - [x] Ribosome surface with 79 chains (40S + 60S)
  - [x] Internal components: mRNA (A4), tRNAs (B4, D4), polypeptide (C4)
- [x] Measure chain centroids and compute animation vectors (`measure_positions.py`)
- [x] Animation script with 5 independent molecule objects (`animate.py`)
  - [x] Separate 6Y0G into component objects (surface, mRNA, tRNA_P, tRNA_A, polypeptide)
  - [x] Keyframe elongation cycle (240 frames: delivery, accommodation, transfer, translocation, departure)
  - [x] Camera orbit (30° over full animation via parented empty)
  - [x] Two-pass render per frame (hide_render toggle)
- [x] Batch compositing script (`composite.py`)
- [x] Video encoding script (`encode.py` — H.264 MP4 + VP9 WebM)
- [x] Debug render test (480x270, 24 frames)
- [ ] Full production render (1920x1080, 240 frames)
- [ ] Deferred to v2:
  - [ ] Pre-build polypeptide chain for progressive reveal
  - [ ] Extend mRNA strand (duplicate + offset)
  - [ ] tRNA rotation animation (v1 is translation only)
  - [ ] Smooth easing curves on keyframes
  - [ ] 3rd tRNA for seamless loop

## Tech stack

- **Python 3.11** — required by Blender
- **Molecular Nodes 4.5.10** — PDB loading, molecular styles
- **bpy (Blender Python)** — headless rendering, materials, keyframes
- **Cycles renderer** — lighting, materials
- **PIL / numpy** — per-frame compositing (translucent overlay + edge outline)
- **ffmpeg** — final video encoding
