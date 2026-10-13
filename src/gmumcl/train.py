"""Generic runner: config -> tasks, model, plugins, Trainer, metrics."""

from __future__ import annotations

from typing import Any, Dict

from .cl.trainer import Trainer, TrainerConfig
from .registry import build_model, build_plugins, build_tasks, scenario_of
from .util.config import set_seed


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run one experiment configuration; returns the metrics dict."""
    set_seed(int(config.get("seed", 0)))
    tasks = build_tasks(config)
    data = config.get("data", {})
    scenario = scenario_of(data)
    growing_head = bool(
        config.get("model", {}).get(
            "growing_head", scenario in ("class_incremental", "task_incremental")
        )
    )
    if growing_head:
        n_classes = max(tasks[0].classes) + 1  # trainer.expand_head grows per task
    else:
        n_classes = max(max(task.classes) for task in tasks) + 1
    model = build_model(config, n_classes)
    plugins = build_plugins(config)
    trainer_section = config.get("trainer", {})
    iterations = trainer_section.get("iterations")
    trainer_config = TrainerConfig(
        epochs=int(trainer_section.get("epochs", 1)),
        lr=float(trainer_section.get("lr", 1e-3)),
        device=str(trainer_section.get("device", "cpu")),
        seed=int(config.get("seed", 0)),
        optimizer=str(trainer_section.get("optimizer", "adam")),
        momentum=float(trainer_section.get("momentum", 0.0)),
        iterations=None if iterations is None else int(iterations),
        verbose=bool(trainer_section.get("verbose", False)),
        task_oracle=scenario == "task_incremental",
    )
    trainer = Trainer(model, plugins, trainer_config)
    matrix = trainer.run(tasks)
    return {
        "method": config.get("method", {}),
        "scenario": scenario,
        "seed": int(config.get("seed", 0)),
        "accuracy_matrix": matrix.rows,
        "average_incremental_accuracy": matrix.average_incremental_accuracy(),
        "forgetting": matrix.forgetting(),
        "backward_transfer": matrix.backward_transfer(),
        "forward_transfer": matrix.forward_transfer(),
        "old_new": list(matrix.final_old_new()),
    }
