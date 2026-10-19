# Scenarios

`data.scenario` selects the task builder; `data.name` selects the dataset.
`rotated_mnist` and `permuted_mnist` imply their scenario. Each builder returns
`TaskData(task_id, classes, fit_loader, calibration_loader, test_loader,
validation_loader=None)`, and the trainer evaluates every task seen so far
after each new task, producing the accuracy matrix `R[t][s]`.

## class_incremental (CI)

Disjoint class groups, one after another; evaluation over all classes seen so
far (no oracle). This is the protocol of `configs/exp/split_mnist_ci.yaml` and
the reproduced table.

```yaml
data: {name: mnist, scenario: class_incremental, classes_per_task: 2}
model: {name: mlp, growing_head: true}
```

## task_incremental (TI)

The same split as CI; the difference is evaluation: `trainer.task_oracle` masks
the logits to the evaluated task's classes, so the classifier may rely on the
task identity.

```yaml
data: {name: mnist, scenario: task_incremental, classes_per_task: 2}
```

## domain_incremental (DI)

Every task carries all classes under a different input transform: rotations for
images (`data.angles`, default evenly spaced over 360 degrees), random shifts
for vector data (`data.shift_scale`). One shared head.

```yaml
data: {name: rotated_mnist, num_tasks: 5}
# or: {name: mnist, scenario: domain_incremental, angles: [0, 90, 180, 270]}
```

## instance_incremental (II)

The dataset is shuffled and cut into `num_tasks` partitions; every task mixes
all classes. One shared head, evaluation over all classes.

```yaml
data: {name: mnist, scenario: instance_incremental, num_tasks: 5}
```

## permuted

Every task applies a fixed random permutation of the flattened inputs; task 0
keeps the identity permutation by default (`data.identity_first: false` to
change). Classes and head are unchanged (Permuted MNIST protocol).

```yaml
data: {name: permuted_mnist, num_tasks: 5}
```

## Calibration and validation split

Every task's training set is split into a fitting subset, a held-out
calibration subset (`data.calibration_fraction`, at least
`data.min_calibration` images) and an optional validation subset
(`data.validation_fraction`, carved out of the fitting images;
`data.calibration_in_fit: true` trains on the calibration images as well while
still exposing them through `calibration_loader`). Baselines train on all data
and set both to 0; calibration-style methods (e.g. AdaReg) keep the split. The
split and the loader shuffles are driven by the config `seed`.

## Datasets

| `data.name` | Loader | Notes |
|---|---|---|
| `toy` | Gaussian class clusters | CPU, no download; `dimension`, `num_classes`, `samples_per_class` |
| `mnist` | torchvision MNIST, 28x28 normalised | `data.root` cache dir |
| `cifar10` | torchvision CIFAR-10 | `data.image_size` resizes (e.g. 224 for ViT) |
| `cifar100` | torchvision CIFAR-100 | `data.image_size` resizes |
| `rotated_mnist` | MNIST + rotations | domain-incremental |
| `permuted_mnist` | MNIST + permutations | permuted |

Any labeled `torch.utils.data.Dataset` (with `targets`/`labels`) works with the
splitters directly; see `examples/custom_method.py`.
