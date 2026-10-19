# Framework

`src/gmumcl/cl/` is a small generic continual-learning framework; methods plug
into it through `cl/plugin.py`. `train.py` + `registry.py` turn a YAML config
into tasks, a model, plugins and metrics.

## Plugin lifecycle

`cl/plugin.py` defines `MethodPluginABC` with the lifecycle calls of a
method:

| Call | When | Typical use |
|---|---|---|
| `set_model(model)` | once | keep the module, install hooks |
| `setup_task(task_id, task)` | before training a task | snapshots, teacher copy, dreamed data |
| `forward(x, y, loss, logits)` | per batch | add the method's loss term |
| `on_after_backward()` | after `backward`, before `step` | path integrals, gradient projection |
| `on_task_end(task_id, task)` | after training a task | Fisher, importances, intervals, projectors |
| `scores(x)` | every evaluation | override the predictor (default: the model's logits) |
| `diagnostics()` | run end | method-specific metrics (config `diagnostics: true`) |
| `figures(output_dir, config, tasks)` | run end | write the method's figures |
| `figure_payload()` | run end | model-free data to redraw figures after selection |

`cl/composer.py` runs `model → criterion → plugins → backward → hooks → step`.
`cl/trainer.py` iterates tasks, expands the head, builds a fresh optimiser per
task (`trainer.optimizer`: Adam by default, SGD with `trainer.momentum`) and
records the accuracy matrices through the trainer's `scores` callable (the
first plugin's `scores`, i.e. the model's logits when it is not overridden).
Training is epoch-based by default; with `trainer.iterations` a task runs
exactly that many optimiser steps, cycling its loader, which is how the
NCL-paper baselines reproduce the paper's iteration budget.

After each task the trainer evaluates every task seen so far and fills
`AccuracyMatrix.rows` (top-1 accuracy), `old_class_rows` (argmax over the
classes seen up to each task), `task_aware_rows` (argmax over the task's own
classes), `new_class_margin_rows` (new-class score margin on old tasks) and —
where the tasks carry a `validation_loader` — `validation_rows`. With
`trainer.task_oracle: true` (set automatically for
`data.scenario: task_incremental`) evaluation masks the scores to the
evaluated task's classes.

## Models

`cl/models.py`: `MLP` (default 2x400 for Split MNIST), `SmallCNN` (no
BatchNorm), `ResNet18` and `ViT`. All expose `expand_head` (grow the head
keeping the learned rows) and `n_classes`. `head_type: rbf` swaps the MLP or
SmallCNN head for `cl/rbf.py:RBFHead`, which Dreaming replays. The module also
has the placement helpers: `protected_layers` (ordered affine layers matching
an fnmatch pattern, head required), `placement_include` (named patterns such
as `head`, `head_and_last_blocks`, `vit_head`) and `paired_batchnorm` (the BN
immediately after an affine layer).

## Scenarios

`cl/scenarios.py` builds class-, task-, domain- and instance-incremental and
permuted tasks from any labeled dataset; see `docs/scenarios.md`. Per task, the
training images are split into a fitting subset (training), a held-out
calibration subset (`calibration_fraction`, at least `min_calibration`) and an
optional validation subset (`validation_fraction`); `calibration_in_fit: true`
also trains on the calibration images. The split and shuffle generators are
seeded from the config `seed`.

## Activation collection

`cl/collect.py` uses the `activationscope` PyPI package with
`capture="input"` (or `"output"`) and `CaptureMode.SNAPSHOT`; it returns one
CPU tensor per named layer, concatenated over a loader. `collect_capped`
streams the loader instead and keeps at most `max_points` rows per layer
(unfolded convolution patches sampled at `patches_per_image` locations), so
patch products stay bounded. activationscope builds
with `-std=c++17`, so install torch 2.8 before building it (see
`docs/usage.md`). Captures that need
gradients (K-FAC output-gradient covariance, NCL factors) stay on raw forward
hooks in `cl/baselines/common.py`, because ActivationScope detaches every
capture.

## Metrics

`cl/metrics.py`: per-task accuracy (optionally restricted to a task's classes)
and the restricted/new-class helpers, all taking a `scores` callable;
`AccuracyMatrix` adds old/new accuracy, old-class, task-aware, new-class
margin and validation matrices, forgetting, backward and forward transfer,
average incremental accuracy and the final accuracies.

## Growth

With `model.growing_head: true` the head starts at the first task's classes and
the trainer appends rows for each next task. Every old-task statistic
(snapshot, importance, Kronecker factor) predates those rows, so old-task
quantities use the leading rows and the rows added for the current task stay
unregularised until its task end (`cl/growth.py`).

## Adding a method

Subclass `MethodPluginABC`, implement whichever calls are needed (override
`scores` for a non-linear predictor, add `diagnostics`/`figures` for reports),
and register it in `cl/baselines/__init__.py`; add a config under
`configs/baselines/` and a test under `tests/unit/` or `tests/component/`.
Every formula needs a unit test against a dense or brute-force reference.
