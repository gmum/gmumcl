"""Unit tests for the reproduction configs (configs/baselines/, configs/exp/)."""

import glob
import os

import yaml

from gmumcl.cl.baselines import REGISTRY

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "baselines")
EXP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "exp")
NCL_PAPER = {"owm", "ncl"}
HSU_ADAPTED = {
    "naive",
    "l2",
    "ewc",
    "ewc_dr",
    "online_ewc",
    "si",
    "mas",
    "lwf",
    "kfac",
    "gpm",
    "sharpening",
    "dreaming",
}


def _configs():
    paths = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.yaml")))
    assert paths, "no baseline configs found"
    for path in paths:
        with open(path) as handle:
            yield path, yaml.safe_load(handle)


def test_every_registered_method_has_a_config():
    configured = {config["method"]["name"] for _, config in _configs()}
    assert configured == set(REGISTRY), configured.symmetric_difference(REGISTRY)


def test_baseline_configs_share_the_split_mnist_model():
    for path, config in _configs():
        assert config["method"]["name"] in REGISTRY, path
        assert config["data"]["name"] == "mnist", path
        assert config["data"]["classes_per_task"] == 2, path
        model = config["model"]
        assert model["name"] == "mlp", path
        assert (model["input_dim"], model["hidden"], model["layers"]) == (784, 400, 2), path


def test_hsu_configs_share_the_adapted_protocol():
    for path, config in _configs():
        if config["method"]["name"] not in HSU_ADAPTED:
            continue
        assert config["data"]["batch_size"] == 128, path
        assert config["model"]["growing_head"] is True, path
        assert config["trainer"]["epochs"] == 4, path
        assert config["trainer"]["lr"] == 0.001, path


def test_ncl_paper_protocol():
    seen = set()
    for path, config in _configs():
        name = config["method"]["name"]
        if name not in NCL_PAPER:
            continue
        seen.add(name)
        assert config["data"]["batch_size"] == 256, path
        assert config["model"]["growing_head"] is False, path
        trainer = config["trainer"]
        assert trainer["optimizer"] == "sgd", path
        assert trainer["momentum"] == 0.9, path
        assert trainer["lr"] == 0.05, path
        assert trainer["iterations"] == 2000, path
    assert seen == NCL_PAPER
    ncl = next(config for path, config in _configs() if path.endswith("ncl.yaml"))
    assert ncl["method"]["data_size"] == 1.0  # unit prior: NCL (no opt)
    assert ncl["method"]["projection_alpha"] == 1e-10
    assert ncl["method"]["lambda"] == 1.0


def test_intact_protocol():
    config = next(config for path, config in _configs() if path.endswith("intact.yaml"))
    assert config["data"]["batch_size"] == 512
    assert config["model"]["growing_head"] is True
    assert config["trainer"]["epochs"] == 5
    assert config["trainer"]["lr"] == 0.0002
    method = config["method"]
    assert method["var_scale"] == 4.2
    assert method["lambda_int_drift"] == 6.0
    assert method["lambda_feat"] == 779.0


def test_naive_is_fine_tuning_without_penalty():
    config = next(config for path, config in _configs() if path.endswith("naive.yaml"))
    assert config["method"]["name"] == "naive"


def test_dreaming_uses_an_rbf_head():
    config = next(config for path, config in _configs() if path.endswith("dreaming.yaml"))
    assert config["model"]["head_type"] == "rbf"


def test_owm_alpha_is_numeric():
    # YAML 1.1 reads 1e-4 as a string; the stabiliser must be parsed as a float.
    config = next(config for path, config in _configs() if path.endswith("owm.yaml"))
    assert isinstance(config["method"]["alpha"], float)
    assert config["method"]["alpha"] == 1e-4


def test_exp_configs_parse_with_known_methods_and_scenarios():
    from gmumcl.registry import SCENARIOS, scenario_of

    for path in sorted(glob.glob(os.path.join(EXP_DIR, "*.yaml"))):
        with open(path) as handle:
            config = yaml.safe_load(handle)
        assert config["method"]["name"] in REGISTRY, path
        assert scenario_of(config["data"]) in SCENARIOS, path
