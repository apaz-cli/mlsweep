"""Guard that the docs shipped inside the wheel stay in sync with the repo docs.

The canonical copies live at the repo root (``README.md`` and ``docs/*.md``) and
are mirrored into ``mlsweep/docs/`` so ``mlsweep docs`` works from an installed
package. This test fails when the two copies drift, so CI will catch it.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    ("README.md", "mlsweep/docs/README.md"),
    ("docs/mlsweep.md", "mlsweep/docs/mlsweep.md"),
    ("docs/sweep_configuration.md", "mlsweep/docs/sweep_configuration.md"),
    ("docs/examples.md", "mlsweep/docs/examples.md"),
]


def test_shipped_docs_match_root_docs():
    for root_rel, shipped_rel in PAIRS:
        root = REPO_ROOT / root_rel
        shipped = REPO_ROOT / shipped_rel
        assert root.is_file(), f"missing {root_rel}"
        assert shipped.is_file(), f"missing {shipped_rel}"
        assert root.read_text() == shipped.read_text(), (
            f"{root_rel} and {shipped_rel} have diverged. Edit the root copy and "
            f"copy it into mlsweep/docs/ (e.g. cp {root_rel} {shipped_rel})."
        )


def test_skill_doc_is_shipped():
    skill = REPO_ROOT / "mlsweep/docs/SKILL.md"
    assert skill.is_file()
    assert "name: mlsweep" in skill.read_text()
