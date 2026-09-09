"""Explicit experiment entry points; run from the repository root."""
import argparse
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Real-corpus AgentPoison, explicit banking demo, ARTEMIS upstream, and ADAPT improvements")
    parser.add_argument("experiment", choices=["agentpoison", "agentpoison-demo", "artemis", "adapt", "gate"])
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the selected runner")
    args = parser.parse_args(argv)
    modules = {
        "agentpoison": "src.agentpoison",
        "agentpoison-demo": "src.agentpoison.run_agentpoison_demo",
        "adapt": "src.adapt",
        "gate": "src.adapt.run_gate_demo",
    }
    upstream = Path(__file__).resolve().parent / "artemis" / "run_pipeline.py"
    if args.experiment == "artemis" and not upstream.is_file():
        parser.error("ARTEMIS upstream is not installed at src/artemis/run_pipeline.py. See _guidance/11_stage0_setup.md.")
    original_argv = sys.argv
    try:
        sys.argv = [args.experiment, *args.args]
        if args.experiment == "artemis":
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
