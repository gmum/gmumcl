"""Run and tabulate the Split-MNIST reproduction configs.

Usage:
    python scripts/reproduce_baselines.py run [method ...] [--seeds 0 1 2]
        [--device cpu] [--outputs DIR]
    python scripts/reproduce_baselines.py table [--outputs DIR]

`run` executes `configs/baselines/<method>.yaml` once per seed and writes
`outputs/baselines/<method>/seed<N>/metrics.json`; `table` prints the
target/reproduced table of `docs/results.md` from those files. The reference
values and protocol are documented in `docs/results.md`.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmumcl import train
from gmumcl.util.config import apply_overrides, load_config

CONFIG_DIR = os.path.join("configs", "baselines")
DEFAULT_OUTPUTS = os.path.join("outputs", "baselines")

# (method key, table label, paper accuracy, paper seeds, source)
ROWS: List[Tuple[str, str, str, str, str]] = [
    ("l2", "L2", "22.52 ± 1.08", "10", "Hsu*"),
    ("ewc", "EWC", "19.80 ± 0.05", "10", "Hsu*"),
    ("online_ewc", "online EWC", "19.77 ± 0.04", "10", "Hsu*"),
    ("si", "SI", "19.67 ± 0.09", "10", "Hsu*"),
    ("mas", "MAS", "19.52 ± 0.29", "10", "Hsu*"),
    ("lwf", "LwF", "24.17 ± 0.33", "10", "Hsu*"),
    ("kfac", "KFAC", "19.99 ± 0.04", "20", "NCL**"),
    ("owm", "OWM", "80.73 ± 1.11", "20", "NCL**"),
    ("ncl", "NCL (no opt)", "47.49 ± 0.84", "20", "NCL**"),
    ("intact", "InTAct", "35.27", "5", "InTAct***"),
    ("naive", "Fine-tune (Adam)", "19.71 ± 0.08", "10", "Hsu*"),
    ("offline", "Offline (upper bound)", "97.53 ± 0.30", "10", "Hsu*"),
]

SUBMITTABLE = [row[0] for row in ROWS if row[0] != "offline"]


def _final_accuracy(metrics: Dict) -> float:
    """Mean accuracy over the five tasks after the last task, in percent."""
    last = metrics["accuracy_matrix"][-1]
    return 100.0 * sum(last) / len(last)


def run(methods: Sequence[str], seeds: Sequence[int], device: str, outputs: str) -> None:
    """One local run per method and seed; writes metrics.json per run."""
    available = sorted(
        os.path.splitext(os.path.basename(path))[0]
        for path in glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
    )
    for method in methods or available:
        config_path = os.path.join(CONFIG_DIR, method + ".yaml")
        if not os.path.exists(config_path):
            raise SystemExit(
                "unknown method %r (no %s); available: %s"
                % (method, config_path, ", ".join(available))
            )
        for seed in seeds:
            config = load_config(config_path)
            apply_overrides(
                config, ["seed=%d" % seed, "trainer.device=%s" % device]
            )
            metrics = train.run(config)
            output_dir = os.path.join(outputs, method, "seed%d" % seed)
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
                json.dump(metrics, handle, indent=2)
            print(
                "%s seed %d: final average accuracy %.2f%%"
                % (method, seed, _final_accuracy(metrics)),
                flush=True,
            )


def seed_values(method: str, outputs: str) -> List[float]:
    """Final accuracies of the finished seeds of one method."""
    pattern = os.path.join(outputs, method, "seed*", "metrics.json")
    values: List[float] = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            values.append(_final_accuracy(json.load(handle)))
    return values


def _cell(values: Sequence[float]) -> str:
    if not values:
        return "pending"
    if len(values) == 1:
        return "%.2f (1 seed)" % values[0]
    return "%.2f ± %.2f" % (statistics.mean(values), statistics.stdev(values))


def render_table(outputs: str) -> str:
    """Markdown table: paper numbers next to the reproduced ones."""
    lines = [
        "| Method | Accuracy (%) | Seeds | Source | Reproduced (%) | Config |",
        "|---|---|---|---|---|---|",
    ]
    for method, label, accuracy, seeds, source in ROWS:
        if method == "offline":
            reproduced, config = "not applicable", "—"
        else:
            reproduced = _cell(seed_values(method, outputs))
            config = "`configs/baselines/%s.yaml`" % method
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (label, accuracy, seeds, source, reproduced, config)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="one local run per method and seed")
    run_parser.add_argument("methods", nargs="*", help="methods to run (default: all)")
    run_parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    run_parser.add_argument("--device", default="cpu", help="trainer.device override")
    run_parser.add_argument("--outputs", default=DEFAULT_OUTPUTS, help="outputs root")

    table_parser = subparsers.add_parser("table", help="print the finished-runs table")
    table_parser.add_argument("--outputs", default=DEFAULT_OUTPUTS, help="outputs root")

    args = parser.parse_args()
    if args.command == "run":
        run(args.methods, args.seeds, args.device, args.outputs)
        return
    print(render_table(args.outputs))


if __name__ == "__main__":
    main()
