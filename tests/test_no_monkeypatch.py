"""Tests take what they need as arguments; none patches the code under test."""

from pathlib import Path

FORBIDDEN = ("monkey" + "patch", "unittest" + ".mock", "mock" + ".patch")


def test_no_test_patches_the_code_under_test():
    offenders = [f"{p.name}: {word}" for p in Path(__file__).parent.rglob("*.py")
                 for word in FORBIDDEN if word in p.read_text()]
    assert offenders == []
