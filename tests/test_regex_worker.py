"""`code search -E` matches in a child process that is killed at its deadline."""

import asyncio
import time

import pytest

from mlsweep._manager_code_read import RegexTimeout, _RegexProcess


def test_matcher_finds_lines():
    async def run():
        deadline = asyncio.get_running_loop().time() + 10
        async with _RegexProcess(r"min\(", 0, deadline) as m:
            assert await m.find(["a", "return min(1, x)", "min (no)"]) == [1]
            assert await m.find([]) == []
    asyncio.run(run())


def test_backtracking_pattern_is_killed_at_the_deadline():
    async def run():
        deadline = asyncio.get_running_loop().time() + 1.0
        async with _RegexProcess(r"(a+)+$", 0, deadline) as m:
            proc = m.proc
            with pytest.raises(RegexTimeout):
                await m.find(["a" * 40 + "b"])
        assert proc.returncode is not None   # killed, not left running

    start = time.monotonic()
    asyncio.run(run())
    assert time.monotonic() - start < 5
