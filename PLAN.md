# Animation Plan

## Goal

~10-15 second looping video of protein translation for a product webpage.
Human 80S ribosome (PDB 6Y0G) with mRNA, tRNAs, and growing polypeptide chain.

## PDB structures

### Primary: 6Y0G — Human 80S ribosome, classical PRE state (3.2 A)

| Chain(s) | Component |
|----------|-----------|
| `S*` (34 chains) | 40S small subunit (18S rRNA + proteins) |
| `L*` (45 chains) | 60S large subunit (28S/5.8S/5S rRNA + proteins) |
| `A4` | mRNA (poly-U, 28 nt) |
| `B4` | P-site tRNA |
| `D4` | A-site tRNA |
| `C4` | Nascent peptide (dipeptide) |

### Style development: 2PTC — Trypsin-BPTI complex (~300 residues)

Small 2-chain structure for fast iteration on materials and shaders.
Chain E = trypsin (transparent surface role), Chain I = BPTI (solid atomic role).

## Animation approach: Rigid-body choreography

Load 6Y0G once, split into component objects (ribosome shell, mRNA, tRNAs, peptide),
and animate them as rigid bodies along scripted keyframe paths. This gives full control
over timing and is much simpler than morphing between PDB states.

### Sequence (one elongation cycle)

```
Frame 0-30:    ESTABLISH — Camera orbits to reveal ribosome with mRNA inside.
               P-site tRNA holds growing peptide. A-site is empty.

Frame 30-90:   DELIVERY — New aminoacyl-tRNA glides into the A-site from outside.
               Enters through the intersubunit space.

Frame 90-120:  ACCOMMODATION — tRNA settles into A-site, slight ribosome flex.

Frame 120-150: PEPTIDE TRANSFER — Peptide chain visually "jumps" from P-site tRNA
               to A-site tRNA (grows by one residue).

Frame 150-210: TRANSLOCATION — tRNAs shift: A→P, P→E. mRNA advances one codon.
               E-site tRNA departs.

Frame 210-240: RESET — Seamless transition back to frame 0 state
               (new tRNA now in P-site, A-site empty, chain is one residue longer).
```

At 24fps this is 10 seconds. The loop should be seamless.

### Camera

Slow continuous orbit (~30 degrees over the full sequence). Slightly angled to show
the exit tunnel where the polypeptide emerges.

## Milestones

- [x] Environment setup (Python 3.11 + Molecular Nodes + headless bpy)
- [x] Basic test render of 6Y0G (proof of pipeline)
- [ ] Nail the visual style on small test structure (2PTC)
  - [ ] Transparent surface + outline shader for "ribosome" role
  - [ ] Opaque vibrant atomic materials for "payload" role
  - [ ] Cel-shaded look with Freestyle or inverted hull
- [ ] Apply style to full 6Y0G ribosome
- [ ] Split 6Y0G into component objects for animation
- [ ] Keyframe the elongation cycle
- [ ] Camera path
- [ ] Render full animation
- [ ] Export as web-ready video (MP4/WebM, H.264/VP9)

## Tech stack

- **Python 3.11** — required by Blender
- **Molecular Nodes 4.5.10** — PDB loading, molecular styles
- **bpy (Blender Python)** — headless rendering, materials, animation
- **Cycles renderer** — for transparency and refraction
- **Freestyle** — for silhouette edge rendering (if Option A for outlines)
