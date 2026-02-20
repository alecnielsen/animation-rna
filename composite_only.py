"""Re-composite from existing pass images (fast iteration on outline params)."""
import numpy as np
from PIL import Image, ImageFilter

OUTLINE_COLOR = (70, 120, 200)
OUTLINE_THICKNESS = 3
SURFACE_OPACITY = 0.20

atoms = Image.open("renders/pass1_internal.png").convert("RGBA")
surface = Image.open("renders/pass2_ribosome_surface.png").convert("RGBA")
surface_gray = Image.open("renders/pass2_ribosome_surface.png").convert("L")

# Layer 1: Translucent surface overlay
surface_np = np.array(surface).astype(np.float32)
surface_np[:, :, 3] = SURFACE_OPACITY * 255
translucent = Image.fromarray(surface_np.astype(np.uint8), "RGBA")
result = Image.alpha_composite(atoms, translucent)

# Layer 2: Outer silhouette from alpha channel (transparent bg render)
alpha = np.array(surface)[:, :, 3]
# Binary mask from alpha
alpha_mask = (alpha > 10).astype(np.uint8) * 255
mask_img = Image.fromarray(alpha_mask)
# Smooth for clean contour
mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=2))
mask_img = Image.fromarray((np.array(mask_img) > 128).astype(np.uint8) * 255)
# Edge detect = clean outer silhouette
silhouette = mask_img.filter(ImageFilter.FIND_EDGES)
sil_np = (np.array(silhouette) > 30).astype(np.uint8) * 255
sil_img = Image.fromarray(sil_np)
for _ in range(OUTLINE_THICKNESS // 2):
    sil_img = sil_img.filter(ImageFilter.MaxFilter(3))

edges_np = np.array(sil_img)
overlay = np.zeros((*edges_np.shape, 4), dtype=np.uint8)
mask = edges_np > 100
overlay[mask, 0] = OUTLINE_COLOR[0]
overlay[mask, 1] = OUTLINE_COLOR[1]
overlay[mask, 2] = OUTLINE_COLOR[2]
overlay[mask, 3] = 255

result = Image.alpha_composite(result, Image.fromarray(overlay, "RGBA"))
result.save("renders/ribosome_style.png")
print("Done! Check renders/ribosome_style.png")
