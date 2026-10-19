"""Generic runner: config -> tasks, model, plugins, Trainer, metrics."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from .cl.trainer import Trainer, TrainerConfig
from .registry import build_model, build_plugins, build_tasks, scenario_of
from .util.config import set_seed


def run(config: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Run one experiment configuration; returns the metrics dict.

    With ``output_dir`` the plugins' figure payloads are saved there and their
    ``figures`` hooks are called; with ``diagnostics: true`` their run
    diagnostics are merged into the metrics.
    """
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
    metrics = {
        "method": config.get("method", {}),
        "scenario": scenario,
        "seed": int(config.get("seed", 0)),
        "accuracy_matrix": matrix.rows,
        "old_class_matrix": matrix.old_class_rows,
        "task_aware_matrix": matrix.task_aware_rows,
        "new_class_margin": matrix.new_class_margin_rows,
        "validation_matrix": matrix.validation_rows,
        "average_incremental_accuracy": matrix.average_incremental_accuracy(),
        "final_average_accuracy": matrix.final_average_accuracy(),
        "validation_final_average_accuracy": matrix.validation_final_average_accuracy(),
        "forgetting": matrix.forgetting(),
        "backward_transfer": matrix.backward_transfer(),
        "forward_transfer": matrix.forward_transfer(),
        "old_new": list(matrix.final_old_new()),
        "final_old_class_accuracy": matrix.final_old_class_accuracy(),
    }
    if output_dir:
        import torch

        payload: Dict[str, Any] = {}
        for plugin in plugins:
            payload.update(plugin.figure_payload())
        if payload:
            torch.save(payload, os.path.join(output_dir, "figure_payload.pt"))
        paths = [path for plugin in plugins for path in plugin.figures(output_dir, config, tasks)]
        if paths:
            metrics["figure_paths"] = paths
    if bool(config.get("diagnostics", False)):
        for plugin in plugins:
            metrics.update(plugin.diagnostics())
    return metrics
