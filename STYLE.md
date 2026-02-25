# Visual Style Specification

## Overview

All molecules rendered as atomic surface meshes (StyleSurface) for a unified, physically realistic look. The ribosome appears as a translucent shell via shader-based transparency, while mRNA, tRNAs, and polypeptide are vibrant and opaque inside. Proper depth occlusion — internal molecules dim/disappear behind the ribosome.

## Rendering approach: Single-pass with shader transparency

Single Cycles render pass. All objects render simultaneously with correct depth.

Ribosome translucency is achieved via Principled BSDF Alpha with HASHED blend mode:
- Alpha = 0.06 on Principled BSDF (stochastic transparency)
- HASHED blend mode provides real translucency without opacity accumulation
- Higher sample counts (64+) smooth out the stochastic noise
- Previous approaches tested and rejected: mix-shader (opacity accumulates over ~50 overlapping layers), Transmission (dark/murky), Fresnel alpha (wireframe look), Glass BSDF (too dark)

No compositing step needed — `animate.py` outputs final frames directly.

## Ribosome (40S + 60S subunits)

- **Representation:** Surface mesh (StyleSurface)
- **Color:** Uniform pale blue-gray, flat diffuse (roughness=1.0)
- **Translucency:** 6% alpha via Principled BSDF (HASHED blend mode, stochastic transparency)
- **Jitter:** Visible rigid-body motion (0.15 BU translation, 5.0deg rotation) via integer-harmonic sum-of-sines
- **No per-chain coloring** — uniform across all subunits

## Internal components (mRNA, tRNAs, polypeptide)

- **Representation:** Surface mesh (StyleSurface) — realistic atomic surface, unified look with ribosome
- **Opacity:** Fully opaque
- **Material:** Principled BSDF, roughness=0.25, emission=0.8
- **Deformation:** Per-residue jitter (coherent displacement per residue) + PCA structural modes

### Per-component colors

| Component | Chain(s) | Color |
|-----------|----------|-------|
| mRNA | A4 | Vibrant blue |
| tRNAs | B4 | Vibrant orange |
| Nascent polypeptide | C4 + extension | Magenta / purple |

## Background

- Dark charcoal (0.04, 0.04, 0.06), low world strength (~0.5)

## Lighting

- Cycles default + world background. Soft, even.
- Emission on internal molecules (strength ~0.8) for self-illumination through translucent ribosome.

## Deformation parameters

```python
# Per-residue jitter (replaces per-atom jitter)
RESIDUE_JITTER_MRNA = 0.05     # BU
RESIDUE_JITTER_TRNA = 0.08     # BU
RESIDUE_JITTER_PEPTIDE = 0.05  # BU

# PCA structural breathing
PCA_BASE_AMP = 1.5  # BU, mode 0 (30:1 ratio over residue jitter)

# Ribosome rigid-body
RIBO_JITTER_TRANS_AMP = 0.15  # BU
RIBO_JITTER_ROT_AMP = 5.0    # degrees
```
