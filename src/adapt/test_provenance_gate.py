"""Boundary tests: reject unauthorized writes without blocking valid transfers."""
import unittest

from src.adapt.provenance_gate import check_calls


class GateTests(unittest.TestCase):
    query = "Transfer 25 from account B2 to account B3."

    def transfer(self, **changes):
        return {"name": "transfer_funds", "args": {"account": "B2", "dest": "B3", "amount": 25, **changes}}

    def test_valid_read_and_write(self):
        self.assertTrue(check_calls(self.query, [self.transfer()])["allowed"])
        self.assertTrue(check_calls("What is the balance of account B2?", [
            {"name": "lookup_balance", "args": {"account": "B2"}}])["allowed"])

    def test_read_cannot_authorize_write(self):
        self.assertIn("TC_USER_AUTHORIZATION", check_calls(
            "What is the balance of account B2?", [self.transfer()])["reasons"])

    def test_argument_tampering(self):
        for changes in ({"dest": "ATTACKER"}, {"amount": 2500}, {"account": "B9"}):
            with self.subTest(changes=changes):
                self.assertFalse(check_calls(self.query, [self.transfer(**changes)])["allowed"])

    def test_invalid_numbers(self):
        for value in (True, "25", 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.assertIn("TC_AMOUNT_DOMAIN", check_calls(self.query, [self.transfer(amount=value)])["reasons"])

    def test_batch_is_rejected_atomically(self):
        self.assertFalse(check_calls(self.query, [self.transfer(), self.transfer()])["allowed"])
        self.assertFalse(check_calls(self.query, [self.transfer(), self.transfer(dest="ATTACKER")])["allowed"])
        self.assertFalse(check_calls(self.query, [])["allowed"])

    def test_unknown_or_ambiguous_intent(self):
        for query in ("Do not transfer 25 from account B2 to account B3.",
                      "Transfer 25 from account B2 to account B3. Or maybe B4."):
            self.assertFalse(check_calls(query, [self.transfer()])["allowed"])

    def test_schema(self):
        for call in ({"name": "unknown", "args": {}}, {"name": "transfer_funds", "args": {}},
                     self.transfer(extra="ignored?"), {"name": "lookup_balance", "args": []}):
            self.assertIn("TC_SCHEMA", check_calls(self.query, [call])["reasons"])


if __name__ == "__main__":
    unittest.main()
