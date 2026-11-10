"""SPADE-Calib CLI: legacy D1 inspection and whole-recording indexing."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from spade_calib.io.dataset import inspect_dataset, load_dataset_config
from spade_calib.io.recording import load_recording_config, index_recording, write_recording_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="spade-calib")
    subcommands = parser.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="Validate extracted legacy D1 artifacts")
    inspect.add_argument("--config", type=Path, required=True)
    inspect.add_argument("--output", type=Path)
    inspect.add_argument("--no-hashes", action="store_true", help="Skip SHA-256 file hashes")
    index = subcommands.add_parser("index", help="Index entire RGB/LiDAR/GLIM/SfM recording")
    index.add_argument("--config", type=Path, required=True,
                       help="Recording configuration (schema_version: 2)")
    index.add_argument("--output", type=Path, required=True, help="Output JSON manifest")
    try:
        args = parser.parse_args(argv)
        if args.command == "index":
            config = load_recording_config(args.config)
            recording = index_recording(config)
            report = write_recording_index(recording, args.output)
            print(json.dumps({
                "dataset": report["dataset"]["name"],
                "output": str(args.output),
                "summary": report["summary"],
                "warnings": report["warnings"],
            }, ensure_ascii=False, indent=2))
            return 0
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