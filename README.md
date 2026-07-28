# MoleculeACE_Re

Reproduction of the [MoleculeACE](https://github.com/molML/MoleculeACE) ECFP
baselines on a modern Python stack, as a shared starting point for
activity-cliff experiments.

The point is not to regenerate numbers that are already published — the official
720-row result table ships with the package. The point is to have a pipeline that
provably stands in the same conditions as those numbers, so a *new* method run
through it can be placed next to them.

## Status — 4 models × 30 targets

| model | RMSE here | RMSE official | Δ | cliff here | cliff official | Δ | agree @4dp | max cell err |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| **SVM** | 0.6712 | 0.6712 | −0.000002 | 0.7511 | 0.7511 | −0.000004 | **30/30 · 29/30** | 0.0001 |
| **GBM** | 0.6953 | 0.6948 | +0.0004 | 0.7836 | 0.7835 | +0.0002 | 0/30 · 0/30 | 0.0519 |
| **RF** | 0.7015 | 0.7000 | +0.0015 | 0.7844 | 0.7857 | −0.0013 | 2/30 · 1/30 | 0.0402 |
| **KNN** | 0.7292 | 0.7290 | +0.0003 | 0.8393 | 0.8388 | +0.0005 | 22/30 · 22/30 | 0.0062 |

Per-target detail: [`results/ecfp_baselines_reproduction.csv`](results/ecfp_baselines_reproduction.csv)

### Only SVM reproduces exactly — and that is expected

The tree ensembles are **not seeded**. `get_benchmark_config` returns
`{'n_estimators': 500}` for RF with no `random_state`, and the package's
`RANDOM_SEED = 42` is not applied to them. Running RF three times on the same
target gives three different numbers:

```
RF   0.665992   0.658890   0.665117     <- differs every run
SVM  0.576209   0.576209   0.576209     <- deterministic
```

**So per-target agreement is not the right check for RF/GBM.** Their means land
within 0.0015 of the official values, which is what a reproduction can claim
without a fixed seed.

**A note worth carrying:** the four means agree to ~0.001 while individual
model×target cells differ by up to 0.05. Aggregate agreement hides per-cell
spread — the same trap the benchmark's own headline numbers set.

If you need bit-exact reproduction of RF/GBM, pass `random_state` explicitly and
report it; the official table cannot be matched cell-for-cell otherwise.

## Setup — this is the part that actually breaks

`pip install MoleculeACE` alone does **not** work. The package's `__init__.py`
begins with `from MoleculeACE.models.mpnn import MPNN`, so **the entire deep
learning stack loads even if you only want an SVM.**

```bash
uv venv --python 3.11 mace-env
uv pip install --python mace-env/bin/python MoleculeACE
uv pip install --python mace-env/bin/python torch torch_geometric transformers tensorflow
uv pip install --python mace-env/bin/python -U rdkit
```

Failures encountered, in order:

| # | Error | Cause |
|---|---|---|
| 1 | `No module named 'torch'` | `__init__` → MPNN |
| 2 | `No module named 'torch_geometric'` | same |
| 3 | `No module named 'transformers'` | `benchmark/utils.py` |
| 4 | `No module named 'tensorflow'` | same |
| 5 | `AttributeError: _ARRAY_API not found` | **numpy 2.4 vs rdkit 2022.09.5 ABI mismatch** |

**(5) is the last hurdle.** The rdkit pulled in by the package is built against
numpy 1; upgrading to rdkit 2026.3.4 clears it.

The upstream README asks for Python 3.8 / TensorFlow 2.9 / PyTorch 1.11 — a 2022
stack. **That is not required.** These numbers were produced on Python 3.11,
torch 2.13, PyG 2.8, transformers 5.14, rdkit 2026.3.4.

## Run

```bash
./mace-env/bin/python scripts/run_ecfp_baselines.py    # SVM, RF, GBM, KNN
./mace-env/bin/python scripts/run_svm_ecfp_30.py       # SVM only
```

Hyperparameters come from `get_benchmark_config`, so there is no tuning step.
SVM takes about 20 s for 30 targets; the ensembles are slower.

## Next

- [ ] Parameterize by target and model
- [ ] Decide whether to fix seeds for RF/GBM, and record the choice
- [ ] Split model coverage across the team

## Reference

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62 (23), 5938–5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
