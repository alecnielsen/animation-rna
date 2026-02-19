# Visual Style Specification

## Overview

Cel-shaded molecular visualization. The ribosome is a ghostly transparent shell that reveals
the solid, vibrant machinery inside.

## Ribosome (40S + 60S subunits)

- **Representation:** Surface mesh only (no atoms, no ribbons, no cartoon)
- **Color:** Uniform single color across all subunits/chains — light cool gray or pale blue
- **Transparency:** Near-transparent (~10-15% opacity). Should be able to clearly see through it
- **Outline:** Solid dark outline on the silhouette/perimeter edges only (Freestyle or inverted-hull)
- **Feel:** Cel-shaded / illustration style. Think "glass ghost" with ink edges

## mRNA (chain A4)

- **Representation:** Atomic (ball-and-stick or spheres)
- **Color:** Vibrant blue
- **Opacity:** Fully opaque
- **Material:** Solid, slightly glossy

## tRNAs (chains B4, D4)

- **Representation:** Atomic (ball-and-stick or spheres)
- **Color:** Vibrant orange (could differentiate A-site vs P-site with orange vs warm yellow)
- **Opacity:** Fully opaque
- **Material:** Solid, slightly glossy

## Nascent polypeptide (chain C4)

- **Representation:** Atomic (ball-and-stick or spheres)
- **Color:** Magenta / purple
- **Opacity:** Fully opaque
- **Material:** Solid, slightly glossy

## Background

- Dark neutral (charcoal or near-black), no gradient

## Lighting

- Soft, even lighting (HDRI or 3-point). No harsh shadows — the focus is readability.

## Technical approach for transparency + outline

The built-in MN `TransparentOutline` material does NOT achieve the desired look (it just dims
the surface while keeping it opaque). We need a custom approach:

### Option A: Principled BSDF with alpha + Freestyle edges
- Custom Principled BSDF material with `alpha = 0.1`, blend mode = alpha blend
- Enable Freestyle rendering in Blender for silhouette edge detection
- Freestyle line set: silhouette + border edges, dark stroke

### Option B: Shader-based outline (inverted hull)
- Two-pass approach: render the surface with a glass/transparent shader
- Add a solidify modifier with flipped normals + black emission material for outline
- More control over line thickness, works in Cycles

### Option C: Compositing
- Render ribosome surface as a separate pass
- Composite with edge detection (Sobel filter) in post
- Most flexible but requires multi-pass pipeline

**Recommendation:** Start with Option A (Principled BSDF + Freestyle). Simplest to script
and Freestyle is well-supported in headless Blender.

## Development strategy

Use a small test structure (e.g. PDB 2PTC — trypsin-BPTI complex, ~300 residues, 2 chains)
to iterate on the shader/outline look before applying to the full 80S ribosome (83 chains,
~300K atoms). This keeps render times under a few seconds per frame.
