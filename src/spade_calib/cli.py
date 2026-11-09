"""Small, read-only CLI for inspecting legacy datasets; no calibration yet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from spade_calib.io.dataset import inspect_dataset, load_dataset_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spade-calib")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="Validate an extracted legacy dataset")
    inspect.add_argument("--config", type=Path, required=True, help="Dataset YAML path")
    inspect.add_argument("--output", type=Path, help="Optional JSON manifest destination")
    inspect.add_argument(
        "--no-hashes", action="store_true", help="Skip SHA-256 input provenance hashes"
    )
    args = parser.parse_args(argv)
    try:
        configuration = load_dataset_config(args.config)
        result = inspect_dataset(configuration, include_hashes=not args.no_hashes)
        rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        print(f"spade-calib: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
