"""Unit tests for the config-driven builders (registry.py)."""

import pytest

from gmumcl.cl.baselines.penalties import Naive
from gmumcl.cl.models import MLP, SmallCNN
from gmumcl.registry import build_model, build_plugins, build_tasks, scenario_of


def _toy_data(**kwargs):
    return {
        "name": "toy",
        "dimension": 6,
        "num_classes": 4,
        "classes_per_task": 2,
        "samples_per_class": 40,
        "calibration_fraction": 0.1,
        "min_calibration": 30,
        "batch_size": 16,
        **kwargs,
    }


def test_build_tasks_dispatches_the_toy_source():
    tasks = build_tasks({"seed": 0, "data": _toy_data()})
    assert [task.classes for task in tasks] == [[0, 1], [2, 3]]


def test_build_tasks_unknown_name_lists_the_sources():
    with pytest.raises(KeyError) as error:
        build_tasks({"data": {"name": "nope"}})
    assert "toy" in str(error.value)


def test_build_tasks_unknown_scenario_lists_the_scenarios():
    with pytest.raises(KeyError) as error:
        build_tasks({"data": _toy_data(scenario="nope")})
    assert "class_incremental" in str(error.value)


def test_build_tasks_forwards_the_config_seed_to_the_split():
    first = build_tasks({"seed": 1, "data": _toy_data()})
    second = build_tasks({"seed": 2, "data": _toy_data()})
    assert first[0].fit_loader.dataset.indices != second[0].fit_loader.dataset.indices


def test_scenario_of_implies_the_scenario_from_dataset_names():
    assert scenario_of({"name": "permuted_mnist"}) == "permuted"
    assert scenario_of({"name": "rotated_mnist"}) == "domain_incremental"
    assert scenario_of({"name": "mnist", "scenario": "task_incremental"}) == "task_incremental"
    assert scenario_of({"name": "toy"}) == "class_incremental"


def test_build_model_dispatches_by_name():
    config = {"data": {"dimension": 6}, "model": {"name": "mlp", "hidden": 8, "layers": 2}}
    model = build_model(config, 4)
    assert isinstance(model, MLP)
    assert model.input_dim == 6
    assert isinstance(build_model({"model": {"name": "small_cnn"}}, 4), SmallCNN)


def test_build_model_supports_the_rbf_head():
    model = build_model({"model": {"name": "mlp", "head_type": "rbf"}}, 4)
    assert type(model.head).__name__ == "RBFHead"


def test_build_model_unknown_name_lists_the_architectures():
    with pytest.raises(KeyError) as error:
        build_model({"model": {"name": "nope"}}, 4)
    assert "mlp" in str(error.value)


def test_build_plugins_dispatches_by_name():
    assert isinstance(build_plugins({"method": {"name": "naive"}})[0], Naive)


def test_build_plugins_unknown_name_lists_the_methods():
    with pytest.raises(KeyError) as error:
        build_plugins({"method": {"name": "nope"}})
    assert "ewc" in str(error.value)
