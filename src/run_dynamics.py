#!/usr/bin/env python

import argparse
import subprocess
import sys
import time
import os
import re
from datetime import datetime

from lct_config import load_lct_input   # adjust import path if needed


def detect_last_step(savedir: str) -> int:
    """
    Detect the last completed step in `savedir` by scanning dict_*.pth
    or occstate_*.txt files.

    Returns:
        last_step (int): the maximum step index found, or -1 if none.
    """
    if not os.path.isdir(savedir):
        return -1

    pattern = re.compile(r"(dict|occstate)_(\d+)\.(pth|txt)$")
    last_step = -1

    for fname in os.listdir(savedir):
        m = pattern.match(fname)
        if m:
            step = int(m.group(2))
            if step > last_step:
                last_step = step

    return last_step


def main():

    parser = argparse.ArgumentParser(
        description="Loop run_step.py to perform LCT dynamics with auto-resume."
    )

    # If start is None, we will auto-detect from savedir
    parser.add_argument("--start", type=int, default=None,
                        help="Start step index. If omitted, auto-detect from savedir.")
    parser.add_argument("--end", type=int, default=None,
                        help="End step index (overrides nsteps in input).")

    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU device ID.")
    parser.add_argument("--input", type=str, default="config/lct.in",
                        help="Path to lct.in file.")
    parser.add_argument("--paths", type=str, default="config/paths.cfg",
                        help="Path to paths.cfg file.")
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter.")

    parser.add_argument("--print_every", type=int, default=50,
                        help="Print progress every N steps.")

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configs: nsteps (from lct.in) and savedir (from paths.cfg)
    # ---------------------------------------------------------
    cfg = load_lct_input(args.input)
    paths_cfg = load_lct_input(args.paths)

    nsteps_from_input = int(cfg.get("nsteps", 0))
    if nsteps_from_input <= 0:
        raise ValueError("nsteps must be defined in lct.in and > 0.")

    savedir = paths_cfg.get("savedir", "./TRAJ_lct_IR")
    os.makedirs(savedir, exist_ok=True)

    # ---------------------------------------------------------
    # Determine end step
    # ---------------------------------------------------------
    if args.end is None:
        end = nsteps_from_input
    else:
        end = args.end

    # ---------------------------------------------------------
    # Determine start step (auto-resume if not provided)
    # ---------------------------------------------------------
    if args.start is None:
        last_done = detect_last_step(savedir)
        start = last_done + 1
    else:
        start = args.start

    if start > end:
        print(f"\n[INFO] start={start} > end={end}. Nothing to do. Exiting.")
        return

    # ---------------------------------------------------------
    # Prepare log file
    # ---------------------------------------------------------
    log_path = os.path.join(savedir, "run_dynamics.log")
    with open(log_path, "a") as log_f:
        log_f.write("\n" + "=" * 60 + "\n")
        log_f.write(f"Run started at {datetime.now().isoformat()}\n")
        log_f.write(f"Input file   : {args.input}\n")
        log_f.write(f"Paths file   : {args.paths}\n")
        log_f.write(f"Savedir      : {savedir}\n")
        log_f.write(f"GPU ID       : {args.gpu_id}\n")
        log_f.write(f"nsteps(input): {nsteps_from_input}\n")
        log_f.write(f"Start step   : {start}\n")
        log_f.write(f"End step     : {end}\n")
        log_f.write("=" * 60 + "\n")

    # ---------------------------------------------------------
    # Display header
    # ---------------------------------------------------------
    print("\n=== LCT Dynamics Loop Start ===")
    print(f"Savedir      : {savedir}")
    print(f"Input nsteps : {nsteps_from_input}")
    print(f"Start step   : {start}")
    print(f"End step     : {end}")
    print(f"GPU ID       : {args.gpu_id}")
    print(f"Input file   : {args.input}")
    print(f"Paths file   : {args.paths}")
    print(f"Log file     : {log_path}\n")

    t0 = time.time()

    # ---------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------
    for nstep in range(start, end + 1):

        percent = (nstep - start) / (end - start + 1) * 100
        print(f"[{percent:6.2f}%] Running step {nstep}/{end}")

        cmd = [
            args.python,
            "run_step.py",
            f"--nstep_record={nstep}",
            f"--gpu_id={args.gpu_id}",
            f"--input={args.input}",
            f"--paths={args.paths}",
        ]

        step_t0 = time.time()
        subprocess.run(cmd, check=True)
        step_elapsed = time.time() - step_t0

        # Periodic progress print
        if (nstep - start) % args.print_every == 0:
            print(f"  Progress: reached step {nstep} "
                  f"(step time {step_elapsed:.2f} s)")

        # Append per-step info to log (lightweight)
        with open(log_path, "a") as log_f:
            log_f.write(
                f"Step {nstep} finished in {step_elapsed:.2f} s "
                f"(total from start {time.time() - t0:.2f} s)\n"
            )

    # ---------------------------------------------------------
    # End
    # ---------------------------------------------------------
    total_elapsed = time.time() - t0
    print("\n=== LCT Dynamics Finished ===")
    print(f"Total elapsed time: {total_elapsed/60:.2f} minutes\n")

    with open(log_path, "a") as log_f:
        log_f.write("-" * 60 + "\n")
        log_f.write(f"Run finished at {datetime.now().isoformat()}\n")
        log_f.write(f"Total elapsed time: {total_elapsed:.2f} s\n")
        log_f.write("-" * 60 + "\n")


if __name__ == "__main__":
    main()
