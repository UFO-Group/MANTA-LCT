# Dependency record

Recorded on 2026-10-13 from the server's working `pysharc` environment. Personal server paths are deliberately omitted from the distributable configuration.

| Component | Observed version / build |
|---|---|
| Python | 3.9.22 (conda-forge) |
| PyTorch | 1.13.1; Conda build `py3.9_cuda11.7_cudnn8.5.0_0` |
| PyTorch CUDA runtime | 11.7 |
| SchNetPack | 1.0.1; installed from a local source directory |
| SchNarc | 0.0.0; installed from a local source directory |
| NumPy | 1.21.6 |
| SciPy | 1.9.3 |
| ASE | 3.26.0 |
| tqdm | 4.67.1 |
| Matplotlib | 3.5.3 |
| h5py | 3.14.0 |
| tensorboardX | 2.6.4 |
| PyYAML | 6.0.2 |
| scikit-learn | 1.6.1 |

`environment.yml` is a focused environment recipe, not an exhaustive lock of every package installed on the server. It does not include unrelated server applications or guarantee identical dependency-solver output. The supplied recipe has not been tested through a fresh environment installation or a new GPU trajectory run.

## Source identity

Both dependency source directories lack Git metadata. Their exact upstream commits cannot be established from the available files, so the label `master` is not used as a reproducible version reference.

The bundled snapshots contain `setup.py`, `README.md`, the original `LICENSE`, and source files. Python package files were taken from the installed environment where available. Original source-tree scripts are also included for packaging; only the package `.py` trees were checked for agreement with the installation.

- **SchNetPack:** all 84 installed package Python files match the server's source tree. Its setup metadata identifies the upstream project as https://github.com/atomistic-machine-learning/schnetpack . No claim of byte-for-byte identity with the public PyPI release is made.
- **SchNarc:** all 9 installed package Python files match the source tree except `calculators.py`, where `hesattr(...)` was corrected to `hasattr(...)`. The snapshot preserves the installed correction and omits the source-only alternate file `schnarc-new.py`, which is absent from the installed package. Distribution version `0.0.0` alone does not identify this source.

The package-tree fingerprint is SHA-256 of sorted relative `.py` paths (UTF-8), each followed by a NUL byte and then the exact file bytes. Paths use `/` separators.

| Installed package tree | SHA-256 |
|---|---|
| schnetpack | `3a2d7b94556cda60ce5de23797b6a108acc143e6796c383678a9ba8174aa493e` |
| schnarc | `97e88928a6dde20d54dcb0a2892011a2430590c5f4e96f45d8e252ccd80f472c` |

Archive checksums and license copies are in `third_party/`. Keeping these snapshots avoids depending on an unpinned upstream checkout. The serialized models require compatible package class definitions; matching package names alone is insufficient.
