"""Patch a base runtime config's ``controller:`` block for one experiment arm.

Usage: python3 scripts/mkconfig.py <base_config.yaml> '<json arm overrides>'

The JSON argument replaces keys of ``controller:`` wholesale (spec, admit,
coordination, spec_kw, admit_kw, coord_kw) -- it is not a deep merge, matching
how run_matrix.sh's ARMS table specifies a complete controller block per arm.
Prints the resulting YAML to stdout; run_matrix.sh redirects it to a file.
"""
from __future__ import annotations

import json
import sys

import yaml


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    base_path, arm_json = argv
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    overrides = json.loads(arm_json)
    cfg["controller"] = overrides

    print(yaml.safe_dump(cfg, sort_keys=False))


if __name__ == "__main__":
    main()
