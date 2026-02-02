# lct_io.py

import os
import torch
import numpy as np


# ============================================================
# I/O for dict_{n}.pth (trajectory data per step)
# ============================================================

def save_step_dict(savedir: str, nstep: int, datadict_tstep: dict):
    """
    Save the per-step data dictionary to:
        {savedir}/dict_{nstep}.pth
    """
    os.makedirs(savedir, exist_ok=True)
    outfile = os.path.join(savedir, f"dict_{nstep}.pth")
    torch.save(datadict_tstep, outfile)


def load_step_dict(savedir: str, nstep: int, device=None):
    """
    Load the per-step data dictionary from:
        {savedir}/dict_{nstep}.pth

    Args:
        savedir (str):
            Output directory containing trajectory data.
        nstep (int):
            Time-step index.
        device (torch.device or None):
            - If None: load without map_location
            - If provided: apply map_location=device

    Returns:
        dict: The stored trajectory dictionary for this time step.
    """
    infile = os.path.join(savedir, f"dict_{nstep}.pth")

    if device is None:
        return torch.load(infile)
    else:
        return torch.load(infile, map_location=device)


# ============================================================
# I/O for occstate_{n}.txt (active electronic state per ring)
# ============================================================

def save_occstate(savedir: str, nstep: int, active_state_index_ca):
    """
    Save the ring-level active-state array to:
        {savedir}/occstate_{nstep}.txt

    Args:
        active_state_index_ca (list or array):
            List of active states for each ring at this time step.
    """
    os.makedirs(savedir, exist_ok=True)
    outfile = os.path.join(savedir, f"occstate_{nstep}.txt")

    arr = np.int64(np.array(active_state_index_ca))
    np.savetxt(outfile, arr, fmt="%d")


def load_occstate(savedir: str, nstep: int):
    """
    Load the ring-level active-state file:
        {savedir}/occstate_{nstep}.txt

    Returns:
        list[int]: active state values for all rings at that time step.
    """
    infile = os.path.join(savedir, f"occstate_{nstep}.txt")

    arr = np.loadtxt(infile, dtype=int)

    # Ensure return type is always a Python list
    return list(np.atleast_1d(arr))
