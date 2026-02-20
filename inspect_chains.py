"""Inspect chain IDs in PDB 6Y0G to determine which are ribosome vs internal components."""
import molecularnodes as mn
import bpy

mn.register()
canvas = mn.Canvas(mn.scene.Cycles(samples=1), resolution=(320, 240))
mol = mn.Molecule.fetch("6Y0G")

# Get the underlying atomic data
arr = mol.array
chains = sorted(set(arr.chain_id))
print(f"\nTotal chains: {len(chains)}")
print(f"All chain IDs: {chains}")

# Categorize
s_chains = [c for c in chains if c.startswith("S")]
l_chains = [c for c in chains if c.startswith("L")]
other = [c for c in chains if not c.startswith("S") and not c.startswith("L")]

print(f"\n40S (S*): {len(s_chains)} chains: {s_chains}")
print(f"60S (L*): {len(l_chains)} chains: {l_chains}")
print(f"Other: {other}")
