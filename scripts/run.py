"""Entry point: python scripts/run.py --config configs/base.yaml -O key=value."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gmumcl import train
from gmumcl.util.config import apply_overrides, load_config, parse_args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args.overrides)
    output_dir = args.output_dir or os.path.join(
        "outputs", os.path.splitext(os.path.basename(args.config))[0]
    )
    os.makedirs(output_dir, exist_ok=True)
    metrics = train.run(config)
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
