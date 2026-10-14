#!/bin/bash
#SBATCH -J IR_T300_5e3
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
mkdir -p TRAJ_new_now_5e3

i=0
while (( i < 1201 )); do
    python -u completest_newLCT_now_Pq_5e3.py  --nstep_record "$i" --NN_id PES --gpu_id 0
    i=$((i+1))
done
