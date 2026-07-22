from __future__ import annotations

import unittest
from pathlib import Path


class NoGripperPostprocessingTest(unittest.TestCase):
    def test_forbidden_terms_absent_from_formal_inference_path(self):
        root = Path(__file__).resolve().parents[2]
        paths = [
            root / "scripts" / "offline_inference.py",
            root / "scripts" / "evaluation" / "compare_chunk_execution.py",
            root / "src" / "soft_vla" / "inference",
        ]
        text = ""
        for path in paths:
            if path.is_dir():
                text += "\n".join(p.read_text(encoding="utf-8") for p in path.rglob("*.py"))
            else:
                text += path.read_text(encoding="utf-8")
        forbidden = ["GripperPostprocessor", "hysteresis", "confirmation_steps", "minimum_hold_steps", "majority_vote"]
        for term in forbidden:
            self.assertNotIn(term, text)


if __name__ == "__main__":
    unittest.main()
