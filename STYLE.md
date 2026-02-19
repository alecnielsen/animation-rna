# Visual Style Specification

## Overview

Translucent molecular surface with solid internal machinery. The ribosome reads as a
semi-transparent shell with an edge outline, while the mRNA, tRNAs, and polypeptide
are vibrant and opaque inside.

## Rendering approach: Two-pass compositing

Single-shader transparency doesn't work on dense molecular surface meshes (opacity
accumulates across thousands of faces). Instead we use two render passes:

1. **Pass 1 (internal):** Cartoon representation of internal components on dark background
2. **Pass 2 (surface):** Opaque flat-shaded surface of the ribosome
3. **Composite in Python (PIL/numpy):**
   - Blend surface over atoms at ~20% opacity (translucent overlay)
   - Edge-detect the alpha-channel silhouette (FIND_EDGES + dilate)
   - Overlay colored outline on top

Camera position is captured from pass 2 (surface is the larger object for framing)
and reused identically in pass 1.

## Ribosome (40S + 60S subunits)

- **Representation:** Surface mesh (StyleSurface)
- **Color:** Uniform pale blue-gray, flat diffuse (roughness=1.0)
- **Translucency:** ~20% opacity via compositing blend
- **Outline:** Blue edge line (~3px), computed via edge detection on alpha-channel silhouette (surface rendered with `film_transparent=True`)
- **No per-chain coloring** — uniform across all subunits

## Internal components (mRNA, tRNAs, polypeptide)

- **Representation:** Cartoon (StyleCartoon) — distinct arrows for beta-sheets, cylinders for helices. Chosen over ribbon (too smooth) and ball-and-stick (too noisy at this scale).
- **Color:** Vibrant blue
- **Opacity:** Fully opaque
- **Material:** Principled BSDF, roughness=0.25, emission=0.8

### Per-component colors

| Component | Chain(s) | Color |
|-----------|----------|-------|
| mRNA | A4 | Vibrant blue |
| tRNAs | B4, D4 | Vibrant orange |
| Nascent polypeptide | C4 + extension | Magenta / purple |

## Background

- Dark charcoal (0.04, 0.04, 0.06), low world strength (~0.5)

## Lighting

- Cycles default + world background. Soft, even.
- Emission on atoms (strength ~0.8) for self-illumination through translucent surface.

## Compositing parameters

```python
SURFACE_OPACITY = 0.20
OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 3       # pixels (MaxFilter dilations)
EDGE_THRESHOLD = 30         # on alpha-channel silhouette
```
