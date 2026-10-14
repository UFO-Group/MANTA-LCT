#!/bin/bash
#SBATCH -J UV_T300_e11
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --error=%j.err
#SBATCH --output=%j.out

set -e
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
mkdir -p TRAJ_pump_300traj_e11

i=0
while (( i < 1201 )); do
    python -u LCT_pump_300traj_e11.py --nstep_record "$i" --NN_id PES --gpu_id 0
    i=$((i+1))
done
