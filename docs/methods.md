# Methods

All methods are `MethodPluginABC` subclasses in `src/gmumcl/cl/baselines/`,
registered by name in `cl/baselines/__init__.py`. Provenance and documented
deviations live in the module docstrings; the reproduced Split-MNIST numbers
and protocol are in `docs/results.md`.

## Penalties (`penalties.py`)

| Name | Loss added | Defaults |
|---|---|---|
| `naive` | — | plain fine-tuning |
| `l2` | `lambda * ||theta - theta*||^2` | `lambda: 1.0` |
| `ewc` | `lambda/2 * sum F (theta - theta*)^2` | `lambda: 1.0`, `fisher_batches: null` |
| `online_ewc` | as EWC with `F <- gamma F + F_task` | `gamma: 1.0` |
| `ewc_dr` | `lambda/2 * sum omega (theta - theta*)^2`, `omega` from reversed logits | `lambda: 1.0`, `omegamax: 1e-4` |
| `si` | `lambda * sum Omega (theta - theta*)^2` | `lambda: 1.0`, `xi: 1e-6` |
| `mas` | `lambda * sum Omega (theta - theta*)^2`, `Omega = |d||f||^2/dtheta|` | `lambda: 1.0` |
| `lwf` | `lambda * task_id * T^2 KL(teacher/T || student/T)` on old-class logits | `lambda: 1.0`, `temperature: 2.0` |

EWC is Kirkpatrick et al. (2017); the diagonal-Fisher implementation follows
gmumcl (MIT) and InTAct-Flex (MIT). EWC-DR is a port of
`scarlet0703/EWC-DR` (commit adcaaeb9): reversed-logit importance averaged over
the task loader and clamped at `omegamax`, merged across tasks with
`alpha = known/total`, halved penalty on the old head rows. SI (Zenke et al.
2017) and MAS (Aljundi et al. 2018) follow the gmumcl ports; LwF (Li and Hoiem
2017) weights the distillation by the number of previous tasks
(`context - 1`), as in `GMvandeVen/continual-learning`.

## Kronecker methods (`kfac.py`, `ncl.py`)

- `kfac` — Kronecker-factored Fisher: input second moments `A` (bias-augmented)
  and sampled-target output-gradient covariances `G`, penalty
  `sum_l tr(G_l dW_l A_l dW_l^T)` from the task-end snapshot. Factor recipe of
  `Xin-Jing/K-FAC` `K_FAC.py` (commit 09f4565c); `lambda: 1.0`,
  `kfac_samples: 256`. `two_weight: true` switches to the proposal's
  two-weight variant (`lambda_center`, `lambda_shape`).
- `ncl` — Natural Continual Learning (Kao et al. 2021), port of
  `tachukao/ncl` (commit 92c2ba5e): per-layer Kronecker priors starting at
  `p_w I` with `p_w = data_size ** -0.5`, additive nearest-Kronecker merge
  after each task, penalty `0.5 sum (delta A) (G delta)`, preconditioned
  updates `P_right g P_left` clipped to norm 10. `lambda: 1.0`,
  `projection_alpha: 1e-10`, `data_size: 1.0`, `grad_clip: 10.0`.

## Projection methods (`projection.py`)

- `owm` — Orthogonal Weight Modification: per affine layer the bias-augmented
  input second moment `A` is averaged over tasks; gradients are projected with
  `P = I - A (A + alpha I)^-1`. Port of `GMvandeVen/continual-learning`
  (commit e6d795aa). `alpha: 1e-10` (paper optimum 1e-4 for class-IL).
- `gpm` — Gradient Projection Memory: task-end representation matrices are
  SVD'd with energy thresholds `threshold_base + threshold_step * task`
  (0.97 + 0.003 t), gradients are projected as `g <- g - (g U) U^T` and
  protected biases are frozen after task 0. Port of `sahagobinda/GPM`
  (commit 1a238ec9). `max_rows: 125`.

## Interval method (`intact.py`)

- `intact` — InTAct (Krukowski et al. 2025), port of `pkrukowski1/InTAct`
  (commit 83b90c55). Every ReLU becomes an `IntervalActivation` recording the
  batch; per-unit 5th/95th percentile intervals merge across tasks by
  min/max. Loss terms: activation variance (`var_scale`), output-preservation
  drift through the next linear layer (`lambda_int_drift`), first-activation
  drift inside the old interval (`lambda_feat`) and a non-overlap alignment
  term. Defaults `var_scale: 0.01`, `lambda_int_drift: 1.0`,
  `lambda_feat: 1.0`, `use_repr_align_loss: true`.

## gmumcl methods

- `sharpening` (`sharpening.py`) — port of `OneAndZero24/local-cl`. A running
  task-average of all affine activations picks the top-K coordinates; the
  current batch's mean activations are pushed towards a sharpened target
  (top coordinates to `gamma * (1 - a)`, the rest scaled by `1 - gamma`).
  Loss `alpha * task_loss + (1 - alpha) * MSE`. Defaults `alpha: 0.5`,
  `gamma: 0.1`, `K: 64`. **Documented deviation:** the original target was
  detached (a constant term with no gradient); here the term is
  differentiable.
- `dreaming` (`dreaming.py`) — port of `OneAndZero24/local-cl`. Needs an RBF
  head (`model.head_type: rbf`). At the start of every task after the first,
  the previous model's per-class kernels (centres and shapes) drive an
  input-space optimisation (`samples_per_class`, `steps`, `dreaming_lr`,
  `lambda_l2`, `lambda_tv`, `diversity_weight`); during the task the current
  model distils the frozen teacher's penultimate activations on the dreamed
  inputs (`alpha`). **Documented deviation:** distillation is the mean squared
  feature error, not the original's unnormalised L2 norm.

## Custom methods

Subclass `MethodPluginABC`, implement the calls you need and register the class
in `cl/baselines/__init__.py`. See `examples/custom_method.py`.
