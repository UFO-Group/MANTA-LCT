# lct_constants.py
import random
import numpy as np
import torch
# ============================================================
# Unit conversion constants
# ============================================================

# Time: 1 fs = 41.34110546116003 atomic units
scale_t = 41.34110546116003

# Mass: 1 amu = 1823.1814419193488 atomic units
scale_m = 1823.1814419193488

# Length: 1 Å = 1.8897259885789233 bohr
scale_L = 1.8897259885789233

# Energy: 1 kJ/mol = 1 / 2625.5 hartree
scale_E = 1.0 / 2625.5




def load_mass(initconds_file: str, natoms: int):
    """
    Load atomic masses from initconds file (6th column, amu).

    This function is a drop-in replacement for:
        np.loadtxt('mass.txt').reshape(Natoms,1) -> tile to (Natoms,3)

    Args:
        initconds_file : str
            Path to initconds file
        natoms : int
            Number of atoms

    Returns:
        Mass_np : ndarray, shape (Natoms, 3)
            Atomic masses in amu, expanded to xyz directions
    """
    with open(initconds_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    # --- find Index 1 ---
    idx_line = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Index"):
            parts = s.split()
            if len(parts) >= 2 and int(parts[1]) == 1:
                idx_line = i
                break
    if idx_line is None:
        raise ValueError("Index 1 not found in initconds file")

    # --- find Atoms ---
    k = idx_line + 1
    while k < len(lines) and lines[k].strip() == "":
        k += 1
    if k >= len(lines) or lines[k].strip() != "Atoms":
        raise ValueError("Atoms line not found after Index")

    atom_start = k + 1
    while atom_start < len(lines) and lines[atom_start].strip() == "":
        atom_start += 1

    # --- read mass column (6th column) ---
    mass_1d = np.zeros((natoms, 1), dtype=float)
    for a in range(natoms):
        row = lines[atom_start + a].split()
        if len(row) < 6:
            raise ValueError(f"Atom line {a+1}: cannot read mass (column < 6)")
        mass_1d[a, 0] = float(row[5])  # 6th column = mass (amu)

    # --- replicate to (Natoms,3), identical to old behavior ---
    Mass_np = np.tile(mass_1d, (1, 3))
    return Mass_np



def load_random_numbers(Nring, Nbead, seed, device):
    """
    Generate reproducible random numbers in [0,1),
    shape: (Nring, Nbead, 1)
    """
    if seed is not None:
        torch.manual_seed(int(seed))
    return torch.rand(Nring, Nbead, 1, device=device)




def read_initconds_sharc(initconds_file, natoms, ntraj, start_index=1):
    """
    Read initconds file and extract coordinates and velocities.

    Args:
        initconds_file: path to initconds file
        natoms: number of atoms
        ntraj: number of trajectories
        start_index: starting Index number

    Returns:
        x_init: (ntraj, natoms, 3) numpy array
        v_init: (ntraj, natoms, 3) numpy array
    """
    with open(initconds_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    index_line = {}
    for ln, line in enumerate(lines):
        s = line.strip()
        if s.startswith("Index"):
            parts = s.split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    index_line[idx] = ln
                except ValueError:
                    pass

    want = list(range(start_index, start_index + ntraj))
    for idx in want:
        if idx not in index_line:
            raise ValueError(f"Missing Index {idx} in initconds file")

    x_all = np.zeros((ntraj, natoms, 3), dtype=float)
    v_all = np.zeros((ntraj, natoms, 3), dtype=float)

    for t, idx in enumerate(want):
        base = index_line[idx]

        k = base + 1
        while k < len(lines) and lines[k].strip() == "":
            k += 1
        if k >= len(lines) or lines[k].strip() != "Atoms":
            raise ValueError(f"Index {idx}: cannot find 'Atoms' line")

        atom_start = k + 1
        while atom_start < len(lines) and lines[atom_start].strip() == "":
            atom_start += 1

        for a in range(natoms):
            row = lines[atom_start + a].split()
            x_all[t, a, 0] = float(row[2])
            x_all[t, a, 1] = float(row[3])
            x_all[t, a, 2] = float(row[4])

            v_all[t, a, 0] = float(row[6])
            v_all[t, a, 1] = float(row[7])
            v_all[t, a, 2] = float(row[8])

    return x_all, v_all
