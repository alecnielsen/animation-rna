"""Build an extended mRNA strand by tiling chain A4 from 6Y0G.

Creates a single continuous ~170-nucleotide mRNA from 10 copies of the
17 nt chain A4, with correct backbone spacing and sequential residue
numbering.  Output: extended_mrna.pdb

Run with: python3.11 build_extended_mrna.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from biotite.structure import AtomArrayStack, concatenate, connect_via_residue_names
from biotite.structure.io.pdb import PDBFile

N_COPIES = 10
CENTER_INDEX = 5  # copy index that stays at crystallographic position
OUTPUT = "extended_mrna.pdb"


def main():
    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    print("=== Building extended mRNA ===")

    # Load 6Y0G and extract chain A4
    mol = mn.Molecule.fetch("6Y0G")
    arr = mol.array
    if isinstance(arr, AtomArrayStack):
        arr = arr[0]

    a4 = arr[arr.chain_id == "A4"]
    print(f"  Chain A4: {len(a4)} atoms")

    # Backbone P atoms for tile offset computation
    p_mask = a4.atom_name == "P"
    p_coords = a4.coord[p_mask]
    print(f"  P atoms: {len(p_coords)}")

    # Tile offset: place each copy so its first P is one backbone step past
    # the previous copy's last P
    step_vec = p_coords[-1] - p_coords[-2]
    tile_offset = (p_coords[-1] + step_vec) - p_coords[0]
    print(f"  Tile offset: ({tile_offset[0]:.1f}, {tile_offset[1]:.1f}, {tile_offset[2]:.1f}) A")
    print(f"  |tile_offset| = {np.linalg.norm(tile_offset):.1f} A")

    # Map each atom's res_id to a 0-based index within the tile
    unique_res = np.unique(a4.res_id)
    n_res = len(unique_res)
    res_to_idx = {int(r): j for j, r in enumerate(unique_res)}
    base_idx = np.array([res_to_idx[int(r)] for r in a4.res_id])
    print(f"  Residues per tile: {n_res}")

    # Create 10 offset copies, centered so i=CENTER_INDEX is at origin
    copies = []
    for i in range(N_COPIES):
        tile = a4.copy()
        tile.coord += (i - CENTER_INDEX) * tile_offset
        tile.res_id = (base_idx + i * n_res + 1).astype(a4.res_id.dtype)
        tile.chain_id[:] = "A"
        copies.append(tile)

    extended = concatenate(copies)
    print(f"  Extended: {len(extended)} atoms, "
          f"{len(np.unique(extended.res_id))} residues")

    # Generate bonds (inter_residue=True connects O3'->P across residue boundaries)
    extended.bonds = connect_via_residue_names(extended, inter_residue=True)

    # Write PDB
    pdb = PDBFile()
    pdb.set_structure(extended)
    pdb.write(OUTPUT)
    print(f"  Written: {OUTPUT}")

    # --- Verification ---
    print(f"\n=== Junction verification (O3'->P at tile boundaries) ===")
    for i in range(N_COPIES - 1):
        last_res = (i + 1) * n_res        # last residue of tile i
        first_res = (i + 1) * n_res + 1   # first residue of tile i+1
        m_o3 = (extended.res_id == last_res) & (extended.atom_name == "O3'")
        m_p = (extended.res_id == first_res) & (extended.atom_name == "P")
        if np.any(m_o3) and np.any(m_p):
            d = np.linalg.norm(extended.coord[m_p][0] - extended.coord[m_o3][0])
            print(f"  Tile {i}->{i+1} (res {last_res}->{first_res}): O3'...P = {d:.2f} A")
        else:
            avail = "O3'" if np.any(m_o3) else "no O3'"
            avail += ", P" if np.any(m_p) else ", no P"
            print(f"  Tile {i}->{i+1} (res {last_res}->{first_res}): {avail}")

    # P-P distances at junctions
    p_ext = extended.coord[extended.atom_name == "P"]
    p_res = extended.res_id[extended.atom_name == "P"]
    print(f"\n=== P-P distances at junctions ===")
    for j in range(len(p_ext) - 1):
        d = np.linalg.norm(p_ext[j + 1] - p_ext[j])
        if p_res[j] % n_res == 0:
            print(f"  P[res {p_res[j]}]->P[res {p_res[j+1]}]: {d:.2f} A (junction)")
        elif abs(d - 5.9) > 3.0:
            print(f"  P[res {p_res[j]}]->P[res {p_res[j+1]}]: {d:.2f} A ** unusual **")

    extent = np.linalg.norm(p_ext[-1] - p_ext[0])
    print(f"\n  Total P-P extent: {extent:.1f} A = {extent * 0.1:.1f} BU")
    print(f"  Total atoms: {len(extended)}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
