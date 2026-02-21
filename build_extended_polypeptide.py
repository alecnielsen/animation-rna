"""Build an extended polyalanine alpha helix for progressive reveal animation.

Creates a ~30-residue polyalanine alpha helix using ideal backbone geometry
(phi=-57, psi=-47), then superimposes the first 2 residues onto chain C4
from 6Y0G (the crystallographic nascent peptide at the peptidyl transferase
center). The helix extends "backwards" through the exit tunnel.

Output: extended_polypeptide.pdb

Run with: python3.11 build_extended_polypeptide.py
"""

import molecularnodes as mn
import bpy
import numpy as np
from biotite.structure import AtomArrayStack, AtomArray, superimpose, BondList
from biotite.structure.io.pdb import PDBFile

N_RESIDUES = 30
OUTPUT = "extended_polypeptide.pdb"

# Ideal alpha helix backbone parameters
PHI = np.radians(-57)
PSI = np.radians(-47)
OMEGA = np.radians(180)  # trans peptide bond

# Bond lengths (Angstroms)
N_CA_LEN = 1.458
CA_C_LEN = 1.523
C_N_LEN = 1.329

# Bond angles
N_CA_C_ANGLE = np.radians(111.0)
CA_C_N_ANGLE = np.radians(116.6)
C_N_CA_ANGLE = np.radians(121.7)


def rotation_matrix(axis, theta):
    """Rodrigues rotation formula: rotate around unit vector axis by angle theta."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def place_atom(prev2, prev1, bond_len, bond_angle, dihedral):
    """Place atom given two predecessors, bond length, angle, and dihedral.

    prev2, prev1: coordinates of two preceding atoms
    bond_len: distance from prev1 to new atom
    bond_angle: angle at prev1 (prev2-prev1-new)
    dihedral: torsion angle around prev2-prev1 bond
    """
    bc = prev1 - prev2
    bc_hat = bc / np.linalg.norm(bc)

    # Initial direction: along bc_hat
    d = bc_hat * bond_len

    # Rotate by -(pi - bond_angle) around an axis perpendicular to bc
    # Find a perpendicular vector
    if abs(bc_hat[0]) < 0.9:
        perp = np.cross(bc_hat, np.array([1, 0, 0]))
    else:
        perp = np.cross(bc_hat, np.array([0, 1, 0]))
    perp = perp / np.linalg.norm(perp)

    R_angle = rotation_matrix(perp, -(np.pi - bond_angle))
    d = R_angle @ d

    # Rotate around bc_hat by dihedral
    R_dihed = rotation_matrix(bc_hat, dihedral)
    d = R_dihed @ d

    return prev1 + d


def build_helix_backbone(n_res):
    """Build ideal alpha helix backbone coordinates for n_res residues.

    Returns dict of {atom_name: (n_res, 3) array} for N, CA, C, O, CB atoms.
    """
    # Seed first two atoms
    coords = {"N": [], "CA": [], "C": [], "O": [], "CB": []}

    # Start: place first N at origin, first CA along +x
    n0 = np.array([0.0, 0.0, 0.0])
    ca0 = np.array([N_CA_LEN, 0.0, 0.0])

    # Place first C using N-CA-C angle and an initial psi guess
    c0 = place_atom(n0, ca0, CA_C_LEN, N_CA_C_ANGLE, PSI)

    coords["N"].append(n0)
    coords["CA"].append(ca0)
    coords["C"].append(c0)

    for i in range(1, n_res):
        prev_ca = coords["CA"][i - 1]
        prev_c = coords["C"][i - 1]
        prev_n = coords["N"][i - 1]

        # N_i: place using CA_{i-1}-C_{i-1} bond, angle CA-C-N, dihedral psi (around CA-C)
        # Actually the dihedral for placing N_i after C_{i-1} is omega (around C-N bond)
        # Sequence: ...-CA_{i-1}-C_{i-1}-N_i-CA_i-C_i-...
        # Dihedral for N_i: N_{i-1}-CA_{i-1}-C_{i-1}-N_i = psi_{i-1}
        n_i = place_atom(prev_ca, prev_c, C_N_LEN, CA_C_N_ANGLE,
                         PSI if i == 1 else PSI)

        # But we need to use the right dihedral context. Let's be more careful:
        # For N_i placement: atoms prev_n, prev_ca, prev_c define the reference.
        # Dihedral N_{i-1}-CA_{i-1}-C_{i-1}-N_i = psi
        n_i = place_atom(prev_ca, prev_c, C_N_LEN, CA_C_N_ANGLE, PSI)

        # CA_i: dihedral CA_{i-1}-C_{i-1}-N_i-CA_i = omega
        ca_i = place_atom(prev_c, n_i, N_CA_LEN, C_N_CA_ANGLE, OMEGA)

        # C_i: dihedral C_{i-1}-N_i-CA_i-C_i = phi
        c_i = place_atom(n_i, ca_i, CA_C_LEN, N_CA_C_ANGLE, PHI)

        coords["N"].append(n_i)
        coords["CA"].append(ca_i)
        coords["C"].append(c_i)

    # Add O and CB for each residue
    for i in range(n_res):
        n_i = coords["N"][i]
        ca_i = coords["CA"][i]
        c_i = coords["C"][i]

        # Carbonyl O: roughly in the peptide plane, opposite to next N
        # Place using CA-C bond with C=O angle ~121 deg, dihedral ~0 relative to N
        o_i = place_atom(ca_i, c_i, 1.231, np.radians(120.5), np.radians(0))
        coords["O"].append(o_i)

        # CB: tetrahedral from CA, opposite to C relative to N-CA
        cb_i = place_atom(c_i, ca_i, 1.521, np.radians(110.1), np.radians(122.6))
        coords["CB"].append(cb_i)

    return {k: np.array(v) for k, v in coords.items()}


def backbone_to_atomarray(backbone, n_res):
    """Convert backbone coordinate dict to biotite AtomArray."""
    atoms_per_res = 5  # N, CA, C, O, CB
    total = n_res * atoms_per_res
    arr = AtomArray(total)

    atom_names = ["N", "CA", "C", "O", "CB"]
    elements = ["N", "C", "C", "O", "C"]

    idx = 0
    for i in range(n_res):
        for j, (aname, elem) in enumerate(zip(atom_names, elements)):
            arr.coord[idx] = backbone[aname][i]
            arr.atom_name[idx] = aname
            arr.res_name[idx] = "ALA"
            arr.res_id[idx] = i + 1
            arr.chain_id[idx] = "A"
            arr.element[idx] = elem
            arr.hetero[idx] = False
            idx += 1

    return arr


def fetch_c4_backbone():
    """Fetch 6Y0G chain C4 and return backbone atoms (N, CA, C)."""
    mn.register()
    mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))

    mol = mn.Molecule.fetch("6Y0G")
    full = mol.array
    if isinstance(full, AtomArrayStack):
        full = full[0]

    c4 = full[full.chain_id == "C4"]
    print(f"  Chain C4: {len(c4)} atoms, {len(np.unique(c4.res_id))} residues")
    print(f"  Residue names: {list(np.unique(c4.res_name))}")

    # Get backbone N, CA, C for superimposition
    bb_mask = np.isin(c4.atom_name, ["N", "CA", "C"])
    bb = c4[bb_mask]
    print(f"  Backbone atoms: {len(bb)}")
    for i, (name, rid, coord) in enumerate(zip(bb.atom_name, bb.res_id, bb.coord)):
        print(f"    {name} res {rid}: ({coord[0]:.1f}, {coord[1]:.1f}, {coord[2]:.1f})")

    return bb


def main():
    print("=== Building extended polypeptide ===")

    # Fetch crystallographic C4 backbone for alignment
    c4_bb = fetch_c4_backbone()

    # Build ideal alpha helix
    print(f"\n  Building {N_RESIDUES}-residue polyalanine alpha helix...")
    backbone = build_helix_backbone(N_RESIDUES)
    helix = backbone_to_atomarray(backbone, N_RESIDUES)

    # Extract first 2 residues' backbone (N, CA, C) from helix for superimposition
    # Match the number of residues in C4
    n_c4_res = len(np.unique(c4_bb.res_id))
    print(f"  C4 has {n_c4_res} residues, using first {n_c4_res} helix residues for alignment")

    helix_bb_mask = np.isin(helix.atom_name, ["N", "CA", "C"]) & (helix.res_id <= n_c4_res)
    helix_bb = helix[helix_bb_mask]

    print(f"  Helix backbone atoms for alignment: {len(helix_bb)}")
    print(f"  C4 backbone atoms for alignment: {len(c4_bb)}")

    if len(helix_bb) != len(c4_bb):
        print(f"  WARNING: atom count mismatch ({len(helix_bb)} vs {len(c4_bb)})")
        # Use minimum common atoms
        n_common = min(len(helix_bb), len(c4_bb))
        helix_bb = helix_bb[:n_common]
        c4_bb = c4_bb[:n_common]

    # Superimpose helix onto C4 position
    fitted, transform = superimpose(c4_bb, helix_bb)
    rmsd = np.sqrt(np.mean(np.sum((fitted.coord - c4_bb.coord) ** 2, axis=1)))
    print(f"  Superimposition RMSD: {rmsd:.2f} A")

    # Apply same transform to full helix
    helix_aligned = helix.copy()
    helix_aligned.coord = transform.apply(helix.coord)

    # Verify alignment
    for rid in [1, 2]:
        ca_mask = (helix_aligned.res_id == rid) & (helix_aligned.atom_name == "CA")
        if np.any(ca_mask):
            ca = helix_aligned.coord[ca_mask][0]
            print(f"  Aligned CA[{rid}]: ({ca[0]:.1f}, {ca[1]:.1f}, {ca[2]:.1f})")

    # Compute helix extent
    ca_coords = helix_aligned.coord[helix_aligned.atom_name == "CA"]
    extent = np.linalg.norm(ca_coords[-1] - ca_coords[0])
    print(f"  Helix CA extent: {extent:.1f} A = {extent * 0.1:.1f} BU")

    # Write PDB
    pdb = PDBFile()
    pdb.set_structure(helix_aligned)
    pdb.write(OUTPUT)
    print(f"  Written: {OUTPUT} ({len(helix_aligned)} atoms, {N_RESIDUES} residues)")
    print("=== Done ===")


if __name__ == "__main__":
    main()
