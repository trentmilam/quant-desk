"""Pytest entry point for the eval.py correctness suite.

Keeps eval.py's existing structure (checks graded against known closed-form values,
not against itself) while making it discoverable/runnable via pytest and standard CI
test reporting.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval as _eval  # noqa: E402


def test_eval_suite_passes():
    assert _eval.main() == 0
