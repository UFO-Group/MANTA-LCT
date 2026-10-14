# MANTA-LCT

MANTA-LCT implements machine-learning-assisted local control theory (ML-LCT) for nonadiabatic molecular dynamics of DBO. It provides infrared (IR) vibrational control and ultraviolet (UV) electronic excitation examples using pretrained potential-energy and dipole-moment models, with nuclear motion represented by five-bead ring polymers.

## Features

### 1. Infrared Vibrational Control

The IR example constructs a control field from the nuclear momentum and dipole derivative along the two C-N bond coordinates to promote bond cleavage on the excited electronic state.

- Driver: `completest_newLCT_now_Pq_5e3.py`
- Initial electronic state: S1
- Control strength: lambda = 0.005 a.u.
- Time step: 0.5 fs

### 2. Ultraviolet Electronic Excitation

The UV example constructs a control field from electronic coefficients and transition dipole moments to drive population transfer from S0 to S1.

- Driver: `LCT_pump_300traj_e11.py`
- Initial electronic state: S0
- Control strength: eta = 11 a.u.
- Time step: 0.05 fs

Both examples use 300 ring polymers, five beads per ring, and a temperature of 300 K. They run independently; the UV output is not automatically passed to the IR example.

## Directory Structure

The repository is organized as follows:

```text
MANTA-LCT/
├── completest_newLCT_now_Pq_5e3.py     # IR dynamics driver
├── LCT_pump_300traj_e11.py            # UV dynamics driver
├── run_IR_T300_5beads_5e3.sh          # IR Slurm submission script
├── run_UV_T300_5beads_e11.sh          # UV Slurm submission script
├── Matrices/
│   ├── electronic.py                 # Electronic-state calculations
│   ├── lct_nuclear_step_RPSH_cayley.py # IR nuclear propagation and control
│   └── lct_nuclear_step_RPSH_Pump.py  # UV nuclear propagation and control
├── Initial_conds_TRPMD_300K/
│   ├── X_0.txt ... X_1499.txt         # Initial nuclear coordinates
│   ├── V_0.txt ... V_1499.txt         # Initial nuclear velocities
│   └── initdb_dict_500.pth            # Neural-network input template
├── New_Model_Ckpt/
│   ├── PES/best_model                # Potential-energy model
│   └── Dipole/best_model             # Dipole-moment model
├── mass.ttx                         # Atomic masses
├── xyzbohr_3dscan_eckart.npy          # Geometry grid
├── random_num_new.npy                # Surface-hopping random-number table
├── third_party/                      # Dependency source snapshots and licenses
├── environment.yml                  # Conda environment recipe
├── requirements-local.txt           # Bundled dependency installation
├── DEPENDENCIES.md                   # Dependency versions and provenance
├── LICENSE                          # MIT license for project software
└── MODEL_DATA_LICENSE.md             # CC BY 4.0 for model weights and data
```

Each `X_i.txt` and `V_i.txt` contains an 18 x 3 array in atomic units. Every five consecutive coordinate/velocity pairs are grouped into one ring polymer. Preserve the file numbering and atom ordering. The 500 rows in `initdb_dict_500.pth` belong to an input template and do not set the number of trajectories.

`Matrices/` also contains historical modules outside the two supported examples. Some still call `KAN(...)` without supplying its implementation or refer to geometry files absent from this repository.

## Installation / Requirements

### 1. Python and Hardware

The recorded working environment uses Linux, Python 3.9.22, and an NVIDIA CUDA GPU. The dynamics drivers require CUDA. Slurm is required to use the supplied submission scripts.

### 2. Required Python Packages

The main dependency versions recorded from the working environment are:

```text
torch          1.13.1 (CUDA 11.7)
schnetpack     1.0.1
schnarc        0.0.0 (bundled source snapshot)
numpy          1.21.6
scipy          1.9.3
ase            3.26.0
tqdm           4.67.1
matplotlib     3.5.3
h5py           3.14.0
tensorboardX   2.6.4
PyYAML         6.0.2
scikit-learn   1.6.1
```

The serialized models require compatible SchNetPack and SchNarc class definitions. Install the bundled source snapshots with the commands below. Their provenance and checksums are documented in [DEPENDENCIES.md](DEPENDENCIES.md) and [third_party/README.md](third_party/README.md).

### 3. Installation

Clone the repository and create the environment:

```bash
git clone https://github.com/UFO-Group/MANTA-LCT.git
cd MANTA-LCT
conda env create -f environment.yml
conda activate dbo-lct
python -m pip install --no-deps -r requirements-local.txt
```

Check CUDA availability and module imports from the repository root:

```bash
python -c "import torch, numpy, scipy, schnetpack, schnarc; print(torch.__version__, torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python -c "from Matrices import lct_nuclear_step_RPSH_Pump, lct_nuclear_step_RPSH_cayley; print('Imports OK')"
```

The environment recipe records the observed server versions; a clean installation and GPU dynamics regression test have not yet been performed with this recipe.

## Usage

Run all commands from the repository root with the environment activated. The programs read their input files using paths relative to this directory.

### 1. Infrared Control

Adjust the partition and GPU resource requests for your cluster, then submit:

```bash
sbatch run_IR_T300_5beads_5e3.sh
```

The default script writes frames 0-1200 to `TRAJ_new_now_5e3/`, corresponding to 600 fs including the initial frame at 0 fs.

### 2. Ultraviolet Control

```bash
sbatch run_UV_T300_5beads_e11.sh
```

The default script writes frames 0-1200 to `TRAJ_pump_300traj_e11/`, corresponding to 60 fs including the initial frame at 0 fs.

Both scripts request one GPU and four CPU cores, run from `$SLURM_SUBMIT_DIR`, and pass `--NN_id PES --gpu_id 0`. `--NN_id PES` selects `New_Model_Ckpt/PES/best_model`; the dipole model is loaded from `New_Model_Ckpt/Dipole/best_model`.

The drivers use `Nsplit_traj=6` and `Nsplit_ca=2` to split model evaluations into batches of 250 beads and 150 centroids. If Hessian evaluation exceeds GPU memory, increase the batch count to a compatible value, such as `Nsplit_traj=12`.

### 3. Outputs and Restarting

Each Python invocation handles one recorded frame:

- `dict_n.pth`: coordinates, velocities, energies, electric fields, electronic coefficients, and other state variables at frame n.
- `occstate_n.txt`: active electronic states of the 300 ring polymers.
- `%j.out` and `%j.err`: Slurm output and error logs.

Frame 0 initializes the simulation. Frame n reads the checkpoint pair from frame n-1. The shell loop sets the run length. To restart after a complete frame 814, change `i=0` to `i=815` in the corresponding submission script. Keep the models, initial conditions, physical parameters, and random-number table consistent with the existing run, and use a separate output directory for each independent simulation.

### 4. Electric-Field Data

The key `E_laser_t_ca` stores electric-field components in atomic units with shape `(300, 1, 3)`. The final dimension is x, y, z. The time of frame n is `n * Delta_t_fs`.

For example, load the initial UV field of the first ring polymer on a CPU:

```python
import torch

data = torch.load('./TRAJ_pump_300traj_e11/dict_0.pth', map_location='cpu')
field_xyz = data['E_laser_t_ca'][0, 0, :]
```

The IR feedback field is shared by all ring polymers. During the initial UV stage, fields differ between rings; a single ring's field is not the ensemble average.

## Reproducibility Notes

- Both drivers use `random_num_new.npy`, with shape `(1000, 10001)`, for surface-hopping decisions. The current ensemble uses the first 300 rows and column n at step n.
- Before 1.2 fs, the UV driver generates startup fields using `0.001 * torch.randn([Nring, 1, 3])`. Each field is shared by the five beads of its ring. The code does not fix the random seed or save the RNG state. From 1.2 fs onward, it uses an ensemble-averaged LCT field.
- The UV example uses a 0.05 fs time step. The manuscript's fitted-UV-pulse simulations use 0.02 fs; these settings should be distinguished when comparing results.
- The supplied inputs are initial conditions for propagation. Their sampling provenance and grouping into five-bead rings still need a complete description. This repository does not include a complete model-training workflow, the QD benchmarks, or dedicated figure-generation scripts.

## Citation

If you use MANTA-LCT, its models, or its data in research, please cite the associated paper. The complete paper reference and DOI will be added when available. Until then, please acknowledge MANTA-LCT authors and contributors and link to [this repository](https://github.com/UFO-Group/MANTA-LCT).

## License

Project-owned software and documentation are distributed under the [MIT License](LICENSE). Project-owned model weights and numerical data are distributed under [CC BY 4.0](MODEL_DATA_LICENSE.md). Both allow commercial use subject to their respective terms. The paper citation is a scholarly request, not an additional license condition. Third-party dependencies retain their original licenses.
