"""Baseline registry: build a plugin by name."""

from __future__ import annotations

from typing import Dict, Optional

from ..plugin import MethodPluginABC
from .dreaming import Dreaming
from .intact import InTAct
from .kfac import KFAC
from .ncl import NCL
from .penalties import EWC, L2, MAS, SI, LwF, Naive
from .projection import GPM, OWM
from .sharpening import Sharpening

REGISTRY: Dict[str, type] = {
    "naive": Naive,
    "l2": L2,
    "ewc": EWC,
    "ewc_dr": EWC,
    "online_ewc": EWC,
    "si": SI,
    "mas": MAS,
    "lwf": LwF,
    "kfac": KFAC,
    "ncl": NCL,
    "owm": OWM,
    "gpm": GPM,
    "intact": InTAct,
    "sharpening": Sharpening,
    "dreaming": Dreaming,
}


def build(name: str, config: Optional[dict] = None) -> MethodPluginABC:
    """Instantiate a baseline by name; config defaults documented per method."""
    if name not in REGISTRY:
        raise KeyError("unknown method %r; available: %s" % (name, sorted(REGISTRY)))
    settings = dict(config or {})
    if name == "ewc_dr":
        settings.setdefault("mode", "ewc_dr")
    elif name == "online_ewc":
        settings.setdefault("mode", "online")
    return REGISTRY[name](settings)
