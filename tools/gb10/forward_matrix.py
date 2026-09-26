#!/usr/bin/env python3
"""Run each configured candidate sequentially on the isolated validation GPU."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path("inference/desktop/lower_policy/configs/policy_config.yaml").read_text())
    pairs = list(config["default_variant_by_skill"].items())
    for overrides in config["variant_sets"].values():
        pairs.extend(overrides.items())
    pairs.append(("rotate_table_base", "rotate_table_base_diffusion"))
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for skill, variant in dict.fromkeys(pairs):
        output = args.output / (variant + ".json")
        command = [sys.executable, str(Path(__file__).with_name("forward_probe.py")),
                   "--skill", skill, "--variant", variant, "--steps", "90", "--output", str(output)]
        print(f"START {skill} {variant}", flush=True)
        started = time.time()
        with output.with_suffix(".log").open("w") as log:
            try:
                rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900).returncode
            except subprocess.TimeoutExpired:
                # A timeout can leave a policy worker alive. Stop the matrix; the
                # operator must inspect/clean it before loading another model.
                rc = 124
        result = {"skill": skill, "variant": variant, "rc": rc,
                  "started_at": started, "seconds": time.time() - started}
        records.append(result)
        (args.output / "matrix.json").write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
