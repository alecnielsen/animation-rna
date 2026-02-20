"""Measure chain centroids from 6Y0G to compute animation offset vectors.

Run with: python3.11 measure_positions.py
"""
import molecularnodes as mn
import bpy
import numpy as np
import math

mn.register()
canvas = mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))
mol = mn.Molecule.fetch("6Y0G")
arr = mol.array

# AtomArrayStack: coord is (1, N, 3), chain_id is (N,)
coords = arr.coord[0]  # first model
chain_ids = arr.chain_id

rot_z = math.pi / 2
cos_r, sin_r = math.cos(rot_z), math.sin(rot_z)

def rotate_z(v):
    x, y, z = v
    return np.array([cos_r * x - sin_r * y, sin_r * x + cos_r * y, z])

def chain_centroid(cid):
    mask = chain_ids == cid
    c = coords[mask].mean(axis=0)
    return c

print("=== Raw PDB centroids (Angstroms) ===")
for cid in ["A4", "B4", "C4", "D4"]:
    c = chain_centroid(cid)
    print(f"  {cid}: ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f})")

print("\n=== Rotated centroids (after 90 deg Z) ===")
centroids = {}
for cid in ["A4", "B4", "C4", "D4"]:
    raw = chain_centroid(cid)
    rotated = rotate_z(raw)
    centroids[cid] = rotated
    print(f"  {cid}: ({rotated[0]:.1f}, {rotated[1]:.1f}, {rotated[2]:.1f})")

p_site = centroids["B4"]
a_site = centroids["D4"]
mrna = centroids["A4"]
peptide = centroids["C4"]

pa_vec = a_site - p_site
print(f"\n=== Offset vectors (Angstroms) ===")
print(f"  P->A vector: ({pa_vec[0]:.1f}, {pa_vec[1]:.1f}, {pa_vec[2]:.1f})")
print(f"  |P->A| = {np.linalg.norm(pa_vec):.1f} A")

e_site = p_site - pa_vec
pe_vec = e_site - p_site
print(f"  P->E vector: ({pe_vec[0]:.1f}, {pe_vec[1]:.1f}, {pe_vec[2]:.1f})")
print(f"  E-site: ({e_site[0]:.1f}, {e_site[1]:.1f}, {e_site[2]:.1f})")

entry = a_site + 2.0 * pa_vec
print(f"  Entry: ({entry[0]:.1f}, {entry[1]:.1f}, {entry[2]:.1f})")

# mRNA principal axis for codon shift
mask_mrna = chain_ids == "A4"
mrna_coords = coords[mask_mrna]
mrna_rotated = np.array([rotate_z(c) for c in mrna_coords])
mrna_centered = mrna_rotated - mrna_rotated.mean(axis=0)
_, _, vt = np.linalg.svd(mrna_centered, full_matrices=False)
mrna_axis = vt[0]
codon_shift = mrna_axis * 10.0  # ~10 A for one codon
print(f"\n  mRNA axis: ({mrna_axis[0]:.3f}, {mrna_axis[1]:.3f}, {mrna_axis[2]:.3f})")
print(f"  Codon shift (10A): ({codon_shift[0]:.1f}, {codon_shift[1]:.1f}, {codon_shift[2]:.1f})")

# Ribosome centroid
ribo_chain_ids = [c for c in sorted(set(chain_ids)) if c.startswith("S") or c.startswith("L")]
mask_ribo = np.isin(chain_ids, ribo_chain_ids)
ribo_coords = coords[mask_ribo]
ribo_rotated = np.array([rotate_z(c) for c in ribo_coords])
ribo_centroid = ribo_rotated.mean(axis=0)
print(f"  Ribosome centroid: ({ribo_centroid[0]:.1f}, {ribo_centroid[1]:.1f}, {ribo_centroid[2]:.1f})")

# Blender objects
print("\n=== Blender objects ===")
for o in sorted(bpy.data.objects, key=lambda x: x.name):
    if "6Y0G" in o.name:
        loc = o.location
        print(f"  {o.name}: loc=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}), type={o.type}")

# Output as constants in nm (MN Blender units = Angstroms / 10)
nm = 0.1
print("\n=== CONSTANTS FOR animate.py (Blender units / nm) ===")
print(f"P_SITE = ({p_site[0]*nm:.2f}, {p_site[1]*nm:.2f}, {p_site[2]*nm:.2f})")
print(f"A_SITE = ({a_site[0]*nm:.2f}, {a_site[1]*nm:.2f}, {a_site[2]*nm:.2f})")
print(f"E_SITE = ({e_site[0]*nm:.2f}, {e_site[1]*nm:.2f}, {e_site[2]*nm:.2f})")
print(f"ENTRY_POS = ({entry[0]*nm:.2f}, {entry[1]*nm:.2f}, {entry[2]*nm:.2f})")
print(f"PA_VEC = ({pa_vec[0]*nm:.2f}, {pa_vec[1]*nm:.2f}, {pa_vec[2]*nm:.2f})")
print(f"CODON_SHIFT = ({codon_shift[0]*nm:.2f}, {codon_shift[1]*nm:.2f}, {codon_shift[2]*nm:.2f})")
print(f"RIBO_CENTROID = ({ribo_centroid[0]*nm:.2f}, {ribo_centroid[1]*nm:.2f}, {ribo_centroid[2]*nm:.2f})")
print(f"PEPTIDE_POS = ({peptide[0]*nm:.2f}, {peptide[1]*nm:.2f}, {peptide[2]*nm:.2f})")
