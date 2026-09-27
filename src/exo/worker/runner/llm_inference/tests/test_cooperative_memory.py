from dataclasses import replace
from typing import Callable, cast

import mlx.core as mx
import pytest

from exo.worker.runner.llm_inference.cooperative_memory import (
    MemoryLimits,
    parse_physical,
    projected_cache,
    reservation_bytes,
)


def test_installed_cpu_allocator_oracle_handles_views_aliases_and_lazy_arrays() -> None:
    measure = cast(Callable[[object], int], cast(object, mx).get_array_buffer_size)
    backing = mx.zeros((1024,), dtype=mx.float32)
    view = backing[:1]
    with pytest.raises(ValueError, match="evaluated"):
        measure(view)
    mx.eval(backing, view)
    allocated = measure(backing)
    assert allocated >= 4096
    assert view.nbytes == 4
    assert measure(view) == allocated
    assert measure([backing, view, backing]) == allocated
    distinct = mx.ones((1024,), dtype=mx.float32)
    mx.eval(distinct)
    assert measure([backing, distinct]) == allocated + measure(distinct)


@pytest.mark.parametrize(
    "tokens,expected_rows", [(1, 256), (256, 256), (257, 512), (262144, 262144)]
)
def test_native_reservation_includes_256_row_floor(
    tokens: int, expected_rows: int
) -> None:
    assert reservation_bytes(tokens - 1, 1) == expected_rows * 65536


def test_warm_large_owner_credit_and_other_owner_future_growth() -> None:
    gib = 1024**3
    # An existing large backing buffer remains charged even if its offset was
    # trimmed. Reusing it for a small request allocates no second large cache.
    projected, growth = projected_cache(
        [0, 4 * gib], [13 * gib, 2 * gib], 0, 16 * 1024**2
    )
    assert projected == 17 * gib
    assert growth == 2 * gib
    assert projected_cache([0, 0], [13 * gib, 0], 0, 16 * 1024**2) == (13 * gib, 0)


def test_darwin_conditional_headroom_ignores_inactive_and_purgeable_credit() -> None:
    text = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Anonymous pages: 100.
Pages wired down: 20.
Pages occupied by compressor: 10.
Pages free: 2.
Pages speculative: 3.
File-backed pages: 40.
Pages inactive: 100.
Pages purgeable: 30.
Swapouts: 7.
"""
    snapshot = parse_physical(
        text, str(200 * 16384), "1", "total = 1.00G used = 0.50M free = 1.00G", 10
    )
    assert snapshot.headroom == 45 * 16384
    assert snapshot.swap_bytes == 512 * 1024
    assert snapshot.healthy(snapshot, 14)
    assert not snapshot.healthy(snapshot, 16)
    assert not replace(snapshot, pressure=2).healthy(snapshot, 10)
    assert not replace(snapshot, swapouts=8).healthy(snapshot, 10)
    assert not replace(snapshot, swap_bytes=512 * 1024 + 1).healthy(snapshot, 10)
    # Do not call the residual total-A all immediately free: cap it by F+E.
    assert replace(snapshot, file_backed=4 * 16384).headroom == 9 * 16384


@pytest.mark.parametrize(
    "capacity,margin",
    [
        (None, "1"),
        ("1", None),
        ("bad", "1"),
        ("1", "bad"),
        ("0", "1"),
        ("1", "-1"),
        ("", "1"),
    ],
)
def test_explicit_invalid_limits_fail_closed(
    monkeypatch: pytest.MonkeyPatch, capacity: str | None, margin: str | None
) -> None:
    for name, value in [
        ("EXO_COOPERATIVE_CACHE_BUDGET_BYTES", capacity),
        ("EXO_COOPERATIVE_PHYSICAL_MARGIN_BYTES", margin),
    ]:
        monkeypatch.delenv(name, raising=False)
        if value is not None:
            monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="invalid_cooperative_memory_configuration"):
        MemoryLimits.configured()


def test_limits_off_only_when_both_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_COOPERATIVE_CACHE_BUDGET_BYTES", raising=False)
    monkeypatch.delenv("EXO_COOPERATIVE_PHYSICAL_MARGIN_BYTES", raising=False)
    assert MemoryLimits.configured() is None
