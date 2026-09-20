"""Offline regression checks for experiment separation and legacy imports."""
import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_default_entrypoint_routes_to_real_corpus(self):
        from src.agentpoison.__main__ import main
        with patch("src.agentpoison.__main__.strategyqa_main") as runner:
            main(["--num-queries", "10"])
            runner.assert_called_once_with(["run", "--num-queries", "10"])
            runner.reset_mock()
            main(["check"])
            runner.assert_called_once_with(["check"])

    def test_phase_entrypoint_routes_to_resumable_runner(self):
        from src.agentpoison.__main__ import main
        with patch("src.agentpoison.__main__.phases_main") as runner:
            main(["ablate", "--plan-only"])
            runner.assert_called_once_with(["ablate", "--plan-only"])

    def test_legacy_module_identity(self):
        aliases = {
            "src.toolpoison.agent": "src.agentpoison.agent",
            "src.toolpoison.memory": "src.agentpoison.memory",
            "src.toolpoison.run_agentpoison_demo": "src.agentpoison.run_agentpoison_demo",
            "src.toolpoison.provenance_gate": "src.adapt.provenance_gate",
            "src.toolpoison.tools": "src.shared.tools",
            "src.toolpoison.tracing": "src.shared.tracing",
            "src.integration.prompt_improve.loop": "src.adapt.extensions.prompt_improve.loop",
            "src.integration.agent_contract.contracts": "src.adapt.extensions.agent_contract.contracts",
            "src.integration.system_testing.graph": "src.adapt.extensions.system_testing.graph",
        }
        for old, new in aliases.items():
            with self.subTest(old=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))

    def test_cli_help_without_model_calls(self):
        commands = [
            ["-m", "src.main", "agentpoison", "--help"],
            ["-m", "src.main", "agentpoison-demo", "--help"],
            ["-m", "src.main", "gate", "--help"],
            ["-m", "src.toolpoison.run_agentpoison_demo", "--help"],
            ["src/toolpoison/run_demo.py", "--help"],
            ["-m", "src.integration.rag_tooling", "--plan"],
            ["-m", "src.main", "adapt", "--plan"],
        ]
        for command in commands:
            with self.subTest(command=command):
                result = subprocess.run([sys.executable, *command], cwd=ROOT,
                                        capture_output=True, text=True, encoding="utf-8", timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.strip())

    def test_attack_does_not_import_defense(self):
        import ast
        for package, forbidden in (
            ("agentpoison", ("src.adapt", "src.integration", "src.toolpoison")),
            ("shared", ("src.adapt", "src.agentpoison", "src.integration", "src.toolpoison")),
            ("adapt", ("src.agentpoison", "src.integration", "src.toolpoison")),
        ):
            for path in (ROOT / "src" / package).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                        [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    for name in names:
                        self.assertFalse(name.startswith(forbidden), f"{path}: {name}")


if __name__ == "__main__":
    unittest.main()
