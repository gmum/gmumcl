# Framework

`src/gmumcl/cl/` is a small generic continual-learning framework; methods plug
into it through `cl/plugin.py`. `train.py` + `registry.py` turn a YAML config
into tasks, a model, plugins and metrics.

## Plugin lifecycle

`cl/plugin.py` defines `MethodPluginABC` with five calls:

| Call | When | Typical use |
|---|---|---|
| `set_model(model)` | once | keep the module, install hooks |
| `setup_task(task_id)` | before training a task | snapshots, teacher copy, dreamed data |
| `forward(x, y, loss, logits)` | per batch | add the method's loss term |
| `on_after_backward()` | after `backward`, before `step` | path integrals, gradient projection |
| `on_task_end(task_id, task)` | after training a task | Fisher, importances, intervals, projectors |

`cl/composer.py` runs `model → criterion → plugins → backward → hooks → step`.
`cl/trainer.py` iterates tasks, expands the head, builds a fresh optimiser per
task (`trainer.optimizer`: Adam by default, SGD with `trainer.momentum`) and
records the accuracy matrix. Training is epoch-based by default; with
`trainer.iterations` a task runs exactly that many optimiser steps, cycling its
loader, which is how the NCL-paper baselines reproduce the paper's iteration
budget.

With `trainer.task_oracle: true` (set automatically for
`data.scenario: task_incremental`) evaluation masks the logits to the
evaluated task's classes.

## Models

`cl/models.py`: `MLP` (default 2x400 for Split MNIST), `SmallCNN` (no
BatchNorm), `ResNet18` and `ViT`. All expose `expand_head` (grow the head
keeping the learned rows) and `n_classes`. `head_type: rbf` swaps the MLP or
SmallCNN head for `cl/rbf.py:RBFHead`, which Dreaming replays.

## Scenarios

`cl/scenarios.py` builds class-, task-, domain- and instance-incremental and
permuted tasks from any labeled dataset; see `docs/scenarios.md`. Per task, the
training images are split into a fitting subset (training) and a held-out
calibration subset (`calibration_fraction`, at least `min_calibration`), with
the split and shuffle generators seeded from the config `seed`.

## Activation collection

`cl/collect.py` uses the `activationscope` PyPI package with
`capture="input"` (or `"output"`) and `CaptureMode.SNAPSHOT`; it returns one
CPU tensor per named layer, concatenated over a loader. Captures that need
gradients (K-FAC output-gradient covariance, NCL factors) stay on raw forward
hooks in `cl/baselines/common.py`, because ActivationScope detaches every
capture.

## Metrics

`cl/metrics.py`: per-task accuracy (optionally restricted to a task's classes)
and `AccuracyMatrix` with old/new accuracy, forgetting, backward and forward
transfer and average incremental accuracy.

## Growth

With `model.growing_head: true` the head starts at the first task's classes and
the trainer appends rows for each next task. Every old-task statistic
(snapshot, importance, Kronecker factor) predates those rows, so old-task
quantities use the leading rows and the rows added for the current task stay
unregularised until its task end (`cl/growth.py`).

## Adding a method

Subclass `MethodPluginABC`, implement whichever calls are needed, and register
it in `cl/baselines/__init__.py`; add a config under `configs/baselines/` and a
test under `tests/unit/` or `tests/component/`. Every formula needs a unit test
against a dense or brute-force reference.
