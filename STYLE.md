# Visual Style Specification

## Overview

Translucent molecular surface with solid internal machinery. The ribosome reads as a
semi-transparent shell with an edge outline, while the mRNA, tRNAs, and polypeptide
are vibrant and opaque inside.

## Rendering approach: Two-pass compositing

Single-shader transparency doesn't work on dense molecular surface meshes (opacity
accumulates across thousands of faces). Instead we use two render passes:

1. **Pass 1 (atoms):** Ball-and-stick of internal components on dark background
2. **Pass 2 (surface):** Opaque flat-shaded surface of the ribosome
3. **Composite in Python (PIL/numpy):**
   - Blend surface over atoms at ~20% opacity (translucent overlay)
   - Edge-detect the surface silhouette (FIND_EDGES + dilate + gaussian blur)
   - Overlay colored outline on top

Camera position is captured from pass 2 (surface is the larger object for framing)
and reused identically in pass 1.

## Ribosome (40S + 60S subunits)

- **Representation:** Surface mesh (StyleSurface)
- **Color:** Uniform pale blue-gray, flat diffuse (roughness=1.0)
- **Translucency:** ~20% opacity via compositing blend
- **Outline:** Blue edge line (~7px), computed via edge detection on the surface silhouette
- **No per-chain coloring** — uniform across all subunits

## mRNA (chain A4)

- **Representation:** Ball-and-stick
- **Color:** Vibrant blue
- **Opacity:** Fully opaque
- **Material:** Principled BSDF, roughness=0.25, emission=0.8

## tRNAs (chains B4, D4)

- **Representation:** Ball-and-stick
- **Color:** Vibrant orange (could differentiate A-site vs P-site)
- **Opacity:** Fully opaque
- **Material:** Same as mRNA

## Nascent polypeptide (chain C4 + procedural extension)

- **Representation:** Ball-and-stick
- **Color:** Magenta / purple
- **Opacity:** Fully opaque
- **Material:** Same as mRNA

## Background

- Dark charcoal (0.04, 0.04, 0.06), low world strength (~0.5)

## Lighting

- Cycles default + world background. Soft, even.
- Emission on atoms (strength ~0.8) for self-illumination through translucent surface.

## Compositing parameters

```python
SURFACE_OPACITY = 0.20
OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 7       # pixels (MaxFilter dilations)
EDGE_THRESHOLD = 15
GAUSSIAN_BLUR = 2.0
```
