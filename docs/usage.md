# Usage

## Install

```bash
pip install gmumcl              # core: torch, numpy, PyYAML, activationscope
pip install "gmumcl[datasets]"  # + torchvision dataset wrappers
pip install "gmumcl[viz]"       # + matplotlib
```

From source: `pip install -e ".[test]"` (needs a C++ toolchain for
activationscope). Python >= 3.10.

## Command line

```bash
python scripts/run.py --config configs/base.yaml
python scripts/run.py --config configs/exp/split_mnist_ci.yaml -O method.name=si -O method.lambda=100 -O trainer.device=cpu
python scripts/run.py --config configs/base.yaml -O data.scenario=task_incremental
```

`-O key.path=value` overrides any key (values parsed with `ast.literal_eval`).
Metrics are printed and written to `outputs/<config-name>/metrics.json`; pass
`--output_dir DIR` to change that.

## Python API

```python
from gmumcl import train

metrics = train.run({
    "seed": 0,
    "data": {"name": "toy", "num_classes": 4, "classes_per_task": 2,
             "calibration_fraction": 0.1, "min_calibration": 20, "batch_size": 32},
    "model": {"name": "mlp", "hidden": 16, "layers": 2, "growing_head": True},
    "trainer": {"epochs": 1, "lr": 0.01, "device": "cpu"},
    "method": {"name": "ewc", "lambda": 1.0},
})
```

`metrics` holds `accuracy_matrix` (`R[t][s]`), `average_incremental_accuracy`,
`forgetting`, `backward_transfer`, `forward_transfer`, `old_new`, `scenario`,
`method` and `seed`.

## Config reference

| Key | Meaning |
|---|---|
| `seed` | seeds parameters, task split and loader shuffles |
| `data.name` | `toy`, `mnist`, `cifar10`, `cifar100`, `rotated_mnist`, `permuted_mnist` |
| `data.scenario` | `class_incremental`, `task_incremental`, `domain_incremental`, `instance_incremental`, `permuted` |
| `data.classes_per_task` / `data.num_tasks` | CI/TI group size / number of tasks for the other scenarios |
| `data.calibration_fraction`, `data.min_calibration` | held-out calibration subset (set 0 to train on all data) |
| `data.batch_size`, `data.root` | loader batch size, dataset cache dir |
| `model.name` | `mlp`, `small_cnn`, `resnet18`, `vit` |
| `model.head_type` | `linear` (default) or `rbf` (Dreaming) |
| `model.growing_head` | grow the head per task (default true for CI/TI) |
| `trainer.epochs` / `trainer.iterations` | epoch budget, or an exact per-task step count |
| `trainer.optimizer`, `trainer.lr`, `trainer.momentum` | optimiser (Adam/SGD) and hyperparameters |
| `trainer.device`, `trainer.verbose` | device string, per-task progress |
| `method.name` | a name from `docs/methods.md`; remaining `method.*` keys go to the plugin |

## Custom components

- **Dataset/model/method names**: extend `DATASET_PAIRS`, `MODEL_BUILDERS` or
  the baseline `REGISTRY` in `src/gmumcl/`.
- **Task list**: call the builders in `gmumcl.cl.scenarios` directly and pass
  the result to `Trainer(model, plugins, TrainerConfig(...)).run(tasks)`.
- **Plugin**: subclass `MethodPluginABC` and register it; see
  `examples/custom_method.py`.
- **Head growth**: models must expose `expand_head(n_classes)` and
  `n_classes`; `cl/growth.py` has the row alignment helpers.

## Tests

```bash
pytest -q                      # CPU, tiny data, no downloads
ruff check src scripts tests
```
