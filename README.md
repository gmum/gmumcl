# gmumcl

[![CI](https://github.com/gmum/gmumcl/actions/workflows/ci.yml/badge.svg)](https://github.com/gmum/gmumcl/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gmumcl)](https://pypi.org/project/gmumcl/)
[![Python](https://img.shields.io/pypi/pyversions/gmumcl)](https://pypi.org/project/gmumcl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A small, generic continual-learning framework for PyTorch: five incremental
scenarios, a five-call plugin API, an accuracy-matrix trainer, and reference
implementations of the standard CL baselines.

## Install

`activationscope` builds a C++ extension with `-std=c++17`, so install torch 2.8
first (newer torch headers require C++20), then build it against that torch:

```bash
pip install "torch==2.8.*" --index-url https://download.pytorch.org/whl/cpu   # or /cu128
pip install setuptools wheel ninja numpy
pip install activationscope --no-build-isolation
pip install gmumcl              # or: pip install "gmumcl[datasets]"
```

From source: `git clone https://github.com/gmum/gmumcl && cd gmumcl && pip install -e ".[test]"`.
Full notes (why the pin, why `--no-build-isolation`): `docs/usage.md`.

## Quickstart

```bash
python scripts/run.py --config configs/base.yaml -O method.name=ewc -O method.lambda=10
```

```python
from gmumcl import train

metrics = train.run({
    "data": {"name": "toy", "num_classes": 4, "classes_per_task": 2},
    "model": {"name": "mlp", "hidden": 16},
    "trainer": {"epochs": 1, "lr": 0.01, "device": "cpu"},
    "method": {"name": "ewc", "lambda": 1.0},
})
print(metrics["accuracy_matrix"])
```

## Scenarios (`data.scenario`)

| Key | Tasks | Evaluation |
|---|---|---|
| `class_incremental` | disjoint class groups | all classes seen so far |
| `task_incremental` | disjoint class groups | task oracle (logits masked to the task) |
| `domain_incremental` | all classes, shifted inputs (rotations) | all classes |
| `instance_incremental` | random partitions of all instances | all classes |
| `permuted` | all classes, one fixed input permutation per task | all classes |

Dataset names: `toy` (CPU clusters), `mnist`, `cifar10`, `cifar100`, plus
`rotated_mnist` (domain-incremental) and `permuted_mnist`.

## Methods (`method.name`)

| Name | Reference | Source |
|---|---|---|
| `naive` | — | plain fine-tuning |
| `l2` | Hsu et al. 2018 | quadratic anchor |
| `ewc` | Kirkpatrick et al. 2017 | gmumcl / InTAct-Flex |
| `ewc_dr` | EWC-DR | `scarlet0703/EWC-DR` (adcaaeb9) |
| `online_ewc` | Schwarz et al. 2018 | gmumcl / InTAct-Flex |
| `si` | Zenke et al. 2017 | gmumcl |
| `mas` | Aljundi et al. 2018 | gmumcl |
| `lwf` | Li & Hoiem 2017 | gmumcl / `GMvandeVen/continual-learning` |
| `kfac` | Kao et al. 2021 | `Xin-Jing/K-FAC` (09f4565c) |
| `ncl` | Kao et al. 2021 | `tachukao/ncl` (92c2ba5e) |
| `owm` | Zeng et al. 2019 | `GMvandeVen/continual-learning` (e6d795aa) |
| `gpm` | Saha et al. 2021 | `sahagobinda/GPM` (1a238ec9) |
| `intact` | Krukowski et al. 2025 | `pkrukowski1/InTAct` (83b90c55) |
| `sharpening` | gmumcl | `OneAndZero24/local-cl` |
| `dreaming` | gmumcl | `OneAndZero24/local-cl` (needs `model.head_type: rbf`) |

## Docs

- `docs/framework.md` — plugin lifecycle, trainer, metrics, adding a method
- `docs/scenarios.md` — the five scenarios and their protocols
- `docs/methods.md` — formulas, defaults and provenance per method
- `docs/usage.md` — install, CLI, configs, custom components
- `docs/results.md` — reproduced Split-MNIST table (from AdaReg) and deviations

## License

MIT (GMUM). See `LICENSE`.
