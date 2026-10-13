# Reproduced results

The Split-MNIST class-incremental table below was reproduced with this exact
framework inside the AdaReg project (`src/adareg/cl/baselines/README.md`,
2025); gmumcl 1.0.0 is that framework extracted and cleaned up, so the
reproduction is the reference for this package too.

Reproduced values are the mean and sample standard deviation of the final
average accuracy (mean over the five test tasks after the last task) over
seeds 0-2, each trained with the configs in `configs/baselines/`.

| Method | Accuracy (%) | Seeds | Source | Reproduced (%) | Config |
|---|---|---|---|---|---|
| L2 | 22.52 ± 1.08 | 10 | Hsu* | 17.96 ± 0.42 | `configs/baselines/l2.yaml` |
| EWC | 19.80 ± 0.05 | 10 | Hsu* | 19.86 ± 0.03 | `configs/baselines/ewc.yaml` |
| online EWC | 19.77 ± 0.04 | 10 | Hsu* | 19.89 ± 0.02 | `configs/baselines/online_ewc.yaml` |
| SI | 19.67 ± 0.09 | 10 | Hsu* | 18.75 ± 0.54 | `configs/baselines/si.yaml` |
| MAS | 19.52 ± 0.29 | 10 | Hsu* | 19.59 ± 0.07 | `configs/baselines/mas.yaml` |
| LwF | 24.17 ± 0.33 | 10 | Hsu* | 19.87 ± 0.04 | `configs/baselines/lwf.yaml` |
| KFAC | 19.99 ± 0.04 | 20 | NCL** | 19.89 ± 0.04 | `configs/baselines/kfac.yaml` |
| OWM | 80.73 ± 1.11 | 20 | NCL** | 83.68 ± 1.37 | `configs/baselines/owm.yaml` |
| NCL (no opt) | 47.49 ± 0.84 | 20 | NCL** | 31.67 ± 4.55 | `configs/baselines/ncl.yaml` |
| InTAct | 35.27 | 5 | InTAct*** | 26.78 ± 1.65 | `configs/baselines/intact.yaml` |
| Fine-tune (Adam) | 19.71 ± 0.08 | 10 | Hsu* | 19.90 ± 0.01 | `configs/baselines/naive.yaml` |
| Offline (upper bound) | 97.53 ± 0.30 | 10 | Hsu* | not applicable | — |

\* Hsu et al., *Re-evaluating Continual Learning Scenarios: A Categorization and
Case for Strong Baselines*, 2018.
\*\* Kao et al., *Natural Continual Learning: Success is a Journey, not (just) a
Destination*, NeurIPS 2021.
\*\*\* Krukowski et al., *InTAct: Interval-based Task Activation Consolidation
for Continual Learning*, 2025.

EWC-DR, GPM, Sharpening and Dreaming have no published Split-MNIST number in
the reference setup; their configs are provided for completeness.

## Protocol

All configs use 5 tasks of 2 contiguous classes, an MLP with two hidden layers
of 400 ReLU units and all training data of each task. Three protocols are
followed, one per target source.

**Hsu-adapted (`l2`, `ewc`, `online_ewc`, `si`, `mas`, `lwf`, `kfac`,
`naive`).** A head grown from 2 to 10 classes (`model.growing_head: true`),
Adam (lr=1e-3), batch 128, 4 epochs per task. The regularisation coefficients
are Hsu et al.'s grid-search optima (L2 100, EWC 600, online EWC 100 with
gamma 1, SI 600 with damping 0.1, MAS 1).

**NCL paper (`owm`, `ncl`).** Fixed 10-way output layer
(`model.growing_head: false`, as in the paper's class-IL setup), batch 256,
2000 iterations per task, SGD with momentum 0.9 and lr 0.05 (Appendix E).
OWM uses the official projector of the NCL feedforward fork's base repo: the
bias-augmented input second moment of each layer, averaged over tasks, with
the fixed stabiliser `alpha = 1e-4` (Table 1). NCL uses the paper's unit
Gaussian prior (`p_w^-2 = 1`, "NCL (no opt)"), projection stabiliser
alpha = 1e-10, lambda = 1, and the full preconditioned Eq. 8 update with the
additive Kronecker prior merge of the feedforward codebase.

**InTAct paper (`intact`).** The same 2x400 MLP, batch 512, 5 epochs, Adam
lr=2e-4, and the published class-IL coefficients (var_scale 4.2,
lambda_int_drift 6, lambda_feat 779).

Documented deviations from the target setups (unchanged from AdaReg):

- Hsu-adapted rows grow the head from the first task's two classes; the rows
  added for the current task stay unregularised until their task end. Hsu's
  scripts pre-allocate a 10-way head but mask the loss the same way. `owm`,
  `ncl` and `intact` are unaffected: the first two keep the paper's fixed
  ten-way head, InTAct's official code grows one as well,
- LwF distills the old-class logits only (rows both models had at the previous
  task end),
- inputs are 28x28 with the standard MNIST normalisation (Hsu pads to 32x32;
  InTAct's selected config uses 32x32),
- the optimiser state (Adam moments or SGD momentum) is re-created for each
  task, while Hsu keeps one optimiser and the official NCL learner carries its
  momentum across tasks,
- NCL is the per-layer feed-forward port of the recurrent official code;
  convolutional factors are averaged over locations, the prior merge is the
  additive nearest-Kronecker sum of the feedforward codebase instead of the
  paper's KL-optimal variant, and transformed gradients are clipped to norm 10,
- OWM is the port of the base repo of the NCL feedforward fork
  (`GMvandeVen/continual-learning`): the projector is built once per task from
  the bias-augmented input second moment and applied to the augmented
  (weight + bias) gradient,
- LwF weights its distillation term by the number of previous tasks
  (`context - 1`), as in that same base repo; a constant lambda is too weak
  once several tasks have been seen,
- InTAct uses our 2x400 MLP on 28x28 inputs; the paper's selected config uses
  a deeper MLP on 32x32 inputs,
- EWC anchors on the last task's snapshot/Fisher (gmumcl/InTAct-Flex port)
  instead of summing all earlier tasks, and its quadratic term is halved;
  online EWC accumulates as in the official implementation,
- SI and MAS use the port's importance normalisations, not Hsu's agents'
  (per-batch unregularised gradients for SI, unnormalised |grad| sums for MAS).

The config `seed` drives both parameter initialisation and the
fitting/calibration split and shuffle generators, so the three seeds differ in
data order too.

## Running

```bash
python scripts/reproduce_baselines.py run ewc owm    # local runs, seeds 0-2
python scripts/reproduce_baselines.py table          # print the table
```

`run` executes `configs/baselines/<method>.yaml` once per seed (override the
device with `-O trainer.device=cpu`, the seeds with `--seeds`) and writes
`outputs/baselines/<method>/seed<N>/metrics.json`; `table` prints the
target/reproduced table from those files.
