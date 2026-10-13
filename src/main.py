"""Explicit experiment entry points; run from the repository root."""
import argparse
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARTEMIS improvements and the AQuA defense pilot")
    parser.add_argument("experiment", choices=["artemis", "artemis-upstream", "gate", "aqua"])
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected runner")
    args = parser.parse_args(argv)
    modules = {
        "artemis": "src.artemis",
        "gate": "src.artemis.run_gate_demo",
        "aqua": "src.defense.aqua",
    }
    upstream = Path(__file__).resolve().parent / "artemis" / "upstream" / "run_pipeline.py"
    if args.experiment == "artemis-upstream" and not upstream.is_file():
        parser.error("ARTEMIS upstream is not installed at src/artemis/upstream/run_pipeline.py. See _guidance/11_stage0_setup.md.")
    original_argv = sys.argv
    try:
        sys.argv = [args.experiment, *args.args]
        if args.experiment == "artemis-upstream":
            sys.path.insert(0, str(upstream.parent))
            try:
                runpy.run_path(str(upstream), run_name="__main__")
            finally:
                sys.path.pop(0)
        else:
            runpy.run_module(modules[args.experiment], run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
