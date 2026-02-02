#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def ask(prompt: str, default=None):
    if default is None:
        return input(f"{prompt}: ").strip()
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else str(default)

def ask_int(prompt: str, default: int):
    while True:
        s = ask(prompt, default)
        try:
            return int(s)
        except ValueError:
            print("  -> Please enter an integer.")

def ask_float(prompt: str, default: float):
    while True:
        s = ask(prompt, default)
        try:
            return float(s)
        except ValueError:
            print("  -> Please enter a float number.")

def ask_path(prompt: str, default: str):
    return ask(prompt, default)

def maybe_quote_path(p: str) -> str:
    p = p.strip()
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        return p
    return f"\"{p}\""

# ======= MODE MENU (numeric input) =======
def choose_lct_mode_numeric():
    """
    Show:
      1. IR
      2. UV
    User inputs 1 or 2.
    """
    print("Select LCT mode:")
    print("  1. IR")
    print("  2. UV")
    while True:
        s = input("Enter 1 or 2 [1]: ").strip()
        if s == "":
            s = "1"
        if s == "1":
            return "IR"
        if s == "2":
            return "UV"
        print("  -> Please enter 1 or 2.")

def write_lct_in(out_path: Path, cfg: dict):
    lines = []
    lines.append("# system")
    lines.append(f"natoms     {cfg['natoms']}")
    lines.append(f"ntraj      {cfg['ntraj']}")
    lines.append(f"nbead      {cfg['nbead']}")
    lines.append(f"nstates    {cfg['nstates']}")
    if cfg["lct"] == "IR":
        lines.append(f"actstate   {cfg['actstate']}")
    lines.append(f"random_seed {cfg['random_seed']}")
    lines.append("")
    lines.append("# time")
    lines.append(f"dt_fs      {cfg['dt_fs']}")
    lines.append(f"nsteps     {cfg['nsteps']}")
    lines.append("")
    lines.append("# field")
    lines.append(f"scale_factor {cfg['scale_factor']}")
    lines.append("")
    lines.append(f"lct        {cfg['lct']}")
    if cfg["lct"] == "IR":
        lines.append(f"id_react   {cfg['id_react']}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_paths_cfg(out_path: Path, cfg: dict):
    lines = []
    lines.append(f"initconds_path   {maybe_quote_path(cfg['initconds_path'])}")
    lines.append(f"topology_file    {maybe_quote_path(cfg['topology_file'])}")
    lines.append(f"topol_file       {maybe_quote_path(cfg['topol_file'])}")
    lines.append(f"pes_model_root   {maybe_quote_path(cfg['pes_model_root'])}")
    lines.append(f"dipole_model_dir {maybe_quote_path(cfg['dipole_model_dir'])}")
    lines.append("")
    lines.append(f"savedir          {maybe_quote_path(cfg['savedir'])}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_submit_sh(out_path: Path, primary_dir: str, python_entry: str):
    content = f"""#!/bin/bash

# $-N lct_job

PRIMARY_DIR={primary_dir}

cd $PRIMARY_DIR

python {python_entry}
"""
    out_path.write_text(content, encoding="utf-8")
    try:
        os.chmod(out_path, 0o755)
    except Exception:
        pass

def create_traj_subfolders(savedir: Path, ntraj: int):
    for i in range(1, ntraj + 1):
        (savedir / f"TRAJ_{i:04d}").mkdir(parents=True, exist_ok=True)

def main():
    print("=== LCT Interactive Input Generator ===")
    print("Press Enter to accept defaults.\n")

    # ---- choose mode first (1/2 menu) ----
    lct_mode = choose_lct_mode_numeric()
    print(f"[Mode] Selected: {lct_mode}\n")

    # ---- defaults (based on your template) ----
    defaults_lct = {
        "natoms": 61,
        "ntraj": 200,
        "nbead": 1,
        "nstates": 2,
        "actstate": 1,      # IR only
        "random_seed": 42,
        "dt_fs": 0.5,
        "nsteps": 6000,
        "scale_factor": 0.0001,
        "id_react": 190,    # IR only
    }
    defaults_paths = {
        "initconds_path": "./Initconds",
        "topology_file": "./Initconds/initdb_dict_300.pth",
        "topol_file": "./Initconds/topol.dat",
        "pes_model_root": "./Model/PES",
        "dipole_model_dir": "./Model/Dipole/",
        "savedir_ir": "./LCT_IR",
        "savedir_uv": "./LCT_UV",
    }

    # ---- ask common LCT inputs ----
    natoms = ask_int("How many atoms (natoms)", defaults_lct["natoms"])
    ntraj  = ask_int("How many trajectories (ntraj)", defaults_lct["ntraj"])
    nbead  = ask_int("How many beads (nbead)", defaults_lct["nbead"])
    nstates = ask_int("How many electronic states (nstates)", defaults_lct["nstates"])

    # ---- ask IR-only parameters (skip for UV) ----
    if lct_mode == "IR":
        actstate = ask_int("Active state index (actstate)", defaults_lct["actstate"])
    else:
        actstate = None

    random_seed = ask_int("Random seed (random_seed)", defaults_lct["random_seed"])

    dt_fs = ask_float("Time step in fs (dt_fs)", defaults_lct["dt_fs"])
    nsteps = ask_int("Number of steps (nsteps)", defaults_lct["nsteps"])
    scale_factor = ask_float("Field scale_factor", defaults_lct["scale_factor"])

    if lct_mode == "IR":
        id_react = ask_int("Reactive mode ID (id_react)", defaults_lct["id_react"])
    else:
        id_react = None

    # ---- ask paths ----
    initconds_path = ask_path("Initconds directory (initconds_path)", defaults_paths["initconds_path"])
    topology_file  = ask_path("Topology file (topology_file)", defaults_paths["topology_file"])
    topol_file     = ask_path("Topol.dat file (topol_file)", defaults_paths["topol_file"])
    pes_model_root = ask_path("PES model root (pes_model_root)", defaults_paths["pes_model_root"])
    dipole_model_dir = ask_path("Dipole model directory (dipole_model_dir)", defaults_paths["dipole_model_dir"])

    savedir_default = defaults_paths["savedir_ir"] if lct_mode == "IR" else defaults_paths["savedir_uv"]
    savedir = ask_path("Main output directory (savedir)", savedir_default)

    # ---- output file names (inside savedir) ----
    lct_in_name = ask("lct.in filename", "lct.in").strip()
    paths_cfg_name = ask("paths.cfg filename", "paths.cfg").strip()

    # ---- create folders ----
    savedir_path = Path(savedir)
    savedir_path.mkdir(parents=True, exist_ok=True)
    create_traj_subfolders(savedir_path, ntraj)

    # ---- write input files under savedir ----
    lct_cfg = {
        "natoms": natoms,
        "ntraj": ntraj,
        "nbead": nbead,
        "nstates": nstates,
        "actstate": actstate,
        "random_seed": random_seed,
        "dt_fs": dt_fs,
        "nsteps": nsteps,
        "scale_factor": scale_factor,
        "lct": lct_mode,
        "id_react": id_react,
    }
    paths_cfg = {
        "initconds_path": initconds_path,
        "topology_file": topology_file,
        "topol_file": topol_file,
        "pes_model_root": pes_model_root,
        "dipole_model_dir": dipole_model_dir,
        "savedir": savedir,
    }

    lct_in_path = savedir_path / lct_in_name
    paths_cfg_path = savedir_path / paths_cfg_name
    write_lct_in(lct_in_path, lct_cfg)
    write_paths_cfg(paths_cfg_path, paths_cfg)

    print(f"\n[OK] Generated: {lct_in_path}")
    print(f"[OK] Generated: {paths_cfg_path}")
    print(f"[OK] Created {ntraj} subfolders under: {savedir_path}")

    # ---- submit.sh (different python entry for IR/UV) ----
    gen_submit = ask("Generate submit.sh? (y/n)", "y").strip().lower()
    if gen_submit == "y":
        primary_dir = ask("PRIMARY_DIR (where to run the job)", str(savedir_path.resolve())).strip()

        if lct_mode == "IR":
            python_entry_default = "./run_dynamics_IR.py"
        else:
            python_entry_default = "./run_dynamics_UV.py"

        python_entry = ask("Python entry script", python_entry_default).strip()
        submit_name = ask("Submit script filename", "submit.sh").strip()

        submit_path = savedir_path / submit_name
        write_submit_sh(submit_path, primary_dir, python_entry)
        print(f"[OK] Generated: {submit_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
