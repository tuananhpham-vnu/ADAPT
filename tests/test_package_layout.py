"""Offline regression checks for experiment separation."""
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PackageLayoutTests(unittest.TestCase):
    def test_cli_help_without_model_calls(self):
        commands = [
            ["-m", "src.main", "gate", "--help"],
            ["-m", "src.artemis.extensions.rag_tooling", "--plan"],
            ["-m", "src.main", "artemis", "--plan"],
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
            ("shared", ("src.artemis", "src.attack", "src.defense")),
            ("artemis", ("src.attack",)),
            ("attack", ("src.defense", "src.artemis")),
        ):
            for path in (ROOT / "src" / package).rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                    names = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                        [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    for name in names:
                        self.assertFalse(name.startswith(forbidden), f"{path}: {name}")


if __name__ == "__main__":
    unittest.main()
