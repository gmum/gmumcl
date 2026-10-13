# Baselines

Reference implementations of the standard continual-learning methods. Every
module docstring names its upstream repo (with the pinned commit), the ported
files and the documented deviations.

| Name | Module | Source |
|---|---|---|
| `naive`, `l2`, `ewc`, `ewc_dr`, `online_ewc`, `si`, `mas`, `lwf` | `penalties.py` | own code (gmumcl / InTAct-Flex), EWC-DR port of `scarlet0703/EWC-DR` (adcaaeb9) |
| `kfac` | `kfac.py` | factor recipe of `Xin-Jing/K-FAC` `K_FAC.py` (09f4565c) |
| `ncl` | `ncl.py` | `tachukao/ncl` (92c2ba5e) |
| `owm`, `gpm` | `projection.py` | `GMvandeVen/continual-learning` (e6d795aa), `sahagobinda/GPM` (1a238ec9) |
| `intact` | `intact.py` | `pkrukowski1/InTAct` (83b90c55) |
| `sharpening`, `dreaming` | `sharpening.py`, `dreaming.py` | `OneAndZero24/local-cl` (MIT) |
| shared helpers | `common.py` | snapshots, diagonal Fisher, Kronecker factors |

Reproduction configs: `configs/baselines/`. The reproduced Split-MNIST table,
protocol and deviations: `docs/results.md`. Add a method by subclassing
`MethodPluginABC` and registering it in `__init__.py`.
