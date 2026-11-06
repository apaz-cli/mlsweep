"""The dashboard's link resolver and file kinds agree with the server's.

The notes tab resolves links in JavaScript so it can do so while you type; the
lint, backlinks and renames use ``_notes.Resolver``.  If they disagree the
dashboard follows links that the lint calls broken and that a rename won't
update.  Both run the same cases from ``notes_links.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from browser import Browser, find_chromium
from campaign_helpers import shared_manager, static_url

from mlsweep import _notes

CASES = json.loads((Path(__file__).parent / "notes_links.json").read_text())
CHROME = find_chromium()


def _python_wiki(src: str, target: str) -> str | None:
    link = _notes.extract_links(f"[[{target}]]")[0]
    _notes.Resolver(CASES["files"]).resolve(src, link)
    return link.resolved


def _python_relative(src: str, href: str) -> str | None:
    links = _notes.extract_links(f"[x]({href})")
    if not links:
        return None
    _notes.Resolver(CASES["files"]).resolve(src, links[0])
    return links[0].resolved


@pytest.mark.parametrize("case", CASES["wiki"], ids=lambda c: f"{c['src']}->{c['target']}")
def test_python_wiki(case: dict[str, str]) -> None:
    assert _python_wiki(case["src"], case["target"]) == case["want"]


@pytest.mark.parametrize("case", CASES["relative"], ids=lambda c: f"{c['src']}->{c['href']}")
def test_python_relative(case: dict[str, str]) -> None:
    assert _python_relative(case["src"], case["href"]) == case["want"]


@pytest.fixture(scope="module")
def page(tmp_path_factory):  # type: ignore[no-untyped-def]
    if CHROME is None:
        pytest.skip("no Chromium")
    for server in shared_manager(tmp_path_factory, "notes_parity"):
        browser = Browser(CHROME)
        browser.goto(static_url(server, "notes.html"))
        browser.wait_for("typeof Resolver !== 'undefined'")
        yield browser
        browser.close()


def _js(page, body: str, data: object):  # type: ignore[no-untyped-def]
    """Run *body* with the fixture's files loaded; one call, so no poll can interleave."""
    files = json.dumps([{"path": p} for p in CASES["files"]])
    return page.eval(f"(() => {{ S.files = {files}; S.fileMap = new Map(); Resolver.rebuild(); "
                     f"const data = {json.dumps(data)}; {body} }})()")


def test_js_wiki(page) -> None:  # type: ignore[no-untyped-def]
    got = _js(page, "return data.map(c => Resolver.wiki(c.src, c.target));", CASES["wiki"])
    assert dict(zip([c["target"] for c in CASES["wiki"]], got)) == {c["target"]: c["want"] for c in CASES["wiki"]}


def test_js_relative(page) -> None:  # type: ignore[no-untyped-def]
    got = _js(page, "return data.map(c => Resolver.relative(c.src, c.href));", CASES["relative"])
    assert [[c["src"], c["href"], g] for c, g in zip(CASES["relative"], got)] == \
        [[c["src"], c["href"], c["want"]] for c in CASES["relative"]]


def test_js_file_kinds_match_server(page) -> None:  # type: ignore[no-untyped-def]
    names = [f"x{ext}" for ext in sorted(_notes.MD_EXT | _notes.IMAGE_EXT | _notes.PDF_EXT | _notes.TEXT_EXT)]
    names += sorted(_notes.TEXT_NAMES) + ["x.unknownext", "noext", "dir/UPPER.PNG"]
    got = _js(page, "return data.map(kindOf);", names)
    assert dict(zip(names, got)) == {n: _notes.kind_of(n) for n in names}


def _moved() -> dict[str, str]:
    return {f: f"moved/{f}" for f in CASES["files"]}


@pytest.mark.parametrize("text,src,want", [(f"[[{c['target']}]]", c["src"], c["want"]) for c in CASES["wiki"]]
                         + [(f"[x]({c['href']})", c["src"], c["want"]) for c in CASES["relative"]])
def test_rename_updates_exactly_the_links_that_resolve(text: str, src: str, want: str | None) -> None:
    """A link the lint and the dashboard follow is one a rename updates, and it still works after."""
    moved = _moved()
    new, count = _notes.rewrite_links(text, src, moved, _notes.Resolver(CASES["files"]))
    assert count == (1 if want else 0)
    if want:
        link = _notes.extract_links(new)[0]
        _notes.Resolver(moved.values()).resolve(src, link)
        assert link.resolved == moved[want]
    else:
        assert new == text
