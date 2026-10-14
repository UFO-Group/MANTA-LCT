# Dependency source snapshots

These small source archives preserve the package code used by the server environment. They are supplied as installation dependencies, not as project-owned source. Their original copyright notices and MIT licenses are retained inside each archive and copied alongside it.

| Archive | Package metadata | License |
|---|---|---|
| `schnetpack-server-snapshot.tar.gz` | SchNetPack 1.0.1 | `SCHNETPACK_LICENSE.txt` |
| `schnarc-server-snapshot.tar.gz` | SchNarc 0.0.0 | `SCHNARC_LICENSE.txt` |

The SchNarc snapshot includes the installed `hesattr` → `hasattr` correction in `calculators.py`. Git commit provenance is unavailable; see `../DEPENDENCIES.md` for the source comparison and package fingerprints.

After creating and activating the environment, install from the repository root:

```bash
python -m pip install --no-deps -r requirements-local.txt
```

`SHA256SUMS` contains archive checksums. This installation path has not yet been validated in a clean environment. The archives contain source only, not the server environment, trained models, datasets, or compiled caches.
