"""Run AgentPoison against the real StrategyQA corpus by default."""
import sys

from src.agentpoison.strategyqa import main as strategyqa_main
from src.agentpoison.phases import main as phases_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    phase_commands = {"optimize", "prepare", "retrieve", "infer", "evaluate", "all", "ablate"}
    if args and args[0] in phase_commands:
        return phases_main(args)
    commands = {"check", "index", "run", "report", "-h", "--help"}
    if not args or args[0] not in commands:
        args.insert(0, "run")
    return strategyqa_main(args)

if __name__ == "__main__":
    main()
