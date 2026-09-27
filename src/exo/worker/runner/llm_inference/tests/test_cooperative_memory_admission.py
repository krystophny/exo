# pyright: reportPrivateUsage=false
import gc
import time
import weakref
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import mlx.core as mx
import pytest
from mlx_lm.models.cache import MLACacheList, QuantizedKVCache
from mlx_lm.models.deepseek_v32 import Model as GlmModel

from exo.shared.types.chunks import ErrorChunk, GenerationChunk
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.llm_inference import cooperative_generator as module
from exo.worker.runner.llm_inference.cooperative_generator import (
    CooperativeGenerator,
    _OwnerPrefixCache,
    _Slot,
)
from exo.worker.runner.llm_inference.cooperative_memory import PhysicalSnapshot
from exo.worker.tests.constants import MODEL_A_ID
from exo.worker.tests.unittests.test_runner.test_cooperative_stream_order import active
from exo.worker.tests.unittests.test_runner.test_event_ordering import CHAT_TASK


def engine(monkeypatch: pytest.MonkeyPatch) -> CooperativeGenerator:
    monkeypatch.setenv("EXO_COOPERATIVE_CACHE_BUDGET_BYTES", str(16 * 1024**3))
    monkeypatch.setenv("EXO_COOPERATIVE_PHYSICAL_MARGIN_BYTES", str(8 * 1024**3))
    monkeypatch.setenv("EXO_KV_BITS", "8")

    def physical() -> PhysicalSnapshot:
        return PhysicalSnapshot(
            captured=time.monotonic(),
            total=64 * 1024**3,
            free=16 * 1024**3,
            file_backed=32 * 1024**3,
            pressure=1,
            valid=True,
        )

    monkeypatch.setattr(module, "capture_physical", physical)
    model = Mock(spec=GlmModel)
    model.args = SimpleNamespace(
        num_hidden_layers=78,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        index_topk_freq=4,
        indexer_types=["full"] * 21 + ["shared"] * 57,
    )
    result = CooperativeGenerator(
        model=cast(Model, model),
        tokenizer=Mock(),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=MODEL_A_ID,
        device_rank=0,
        cancel_receiver=Mock(),
        event_sender=Mock(),
    )
    monkeypatch.setattr(result, "agree_on_tasks", Mock())
    monkeypatch.setattr(result, "agree_on_cancellations", Mock())
    return result


def cached(rows: int = 256) -> list[MLACacheList]:
    cache = QuantizedKVCache(group_size=64, bits=8)
    update = cast(
        Callable[
            [mx.array, mx.array],
            tuple[
                tuple[mx.array, mx.array, mx.array], tuple[mx.array, mx.array, mx.array]
            ],
        ],
        cache.update_and_fetch,
    )
    quantized = update(
        mx.zeros((1, 1, rows, 512), dtype=mx.bfloat16),
        mx.zeros((1, 1, rows, 64), dtype=mx.bfloat16),
    )
    mx.eval(quantized)
    return [MLACacheList(cache)]


def test_warm_prefix_buffers_are_credited_and_cross_owner_alias_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = engine(monkeypatch)
    prefix = cast(_OwnerPrefixCache, result._slots[0].kv_prefix_cache)
    cache = cached()
    prefix.caches.append(cache)
    allocated, nonoverlap = result._owner_buffers()
    assert nonoverlap and allocated[0] > 0
    # A small request reuses the warm storage without adding another full budget.
    assert result._admit_memory(0, 65536, list(result._slots))
    assert prefix.caches[0] is cache
    other = cast(_OwnerPrefixCache, result._slots[1].kv_prefix_cache)
    other.caches.append(cache)
    _, nonoverlap = result._owner_buffers()
    assert not nonoverlap


@pytest.mark.parametrize("rank", [0, 1])
def test_asymmetric_pressure_has_matching_collectives_and_only_idle_eviction(
    monkeypatch: pytest.MonkeyPatch, rank: int
) -> None:
    result = engine(monkeypatch)
    active_slot, idle_slot = result._slots
    owner = CHAT_TASK
    active_slot._active = active(owner)
    active_identity = active_slot._active
    cache = cached()
    active_prefix = cast(_OwnerPrefixCache, active_slot.kv_prefix_cache)
    idle_prefix = cast(_OwnerPrefixCache, idle_slot.kv_prefix_cache)
    active_prefix.active_cache = cache
    idle_prefix.caches.append(cached())
    active_buffers = result._owner_buffers()[0][0]
    gathered: list[tuple[int, int, int]] = []

    def rows(chosen: int, requested: int, idle_mask: int) -> list[list[int]]:
        gathered.append((chosen, requested, idle_mask))
        local = [1, idle_mask, chosen, requested, 0, 0, 0, 0, 0]
        peer = list(local)
        if len(gathered) == 1:
            peer[0] = 0
        return [local, peer] if rank == 0 else [peer, local]

    monkeypatch.setattr(result, "_memory_rows", rows)
    assert result._admit_memory(1, 65536, [idle_slot])
    assert 1 <= len(gathered) <= 3 and all(row == gathered[0] for row in gathered)
    assert len(idle_prefix.caches) == 1
    assert active_slot._active is active_identity
    assert active_prefix.active_cache is cache
    assert result._owner_buffers()[0][0] == active_buffers


def test_failed_kernel_probe_is_unsafe_and_still_enters_fixed_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = engine(monkeypatch)
    monkeypatch.setattr(module, "capture_physical", lambda: PhysicalSnapshot())
    result.group = Mock()
    cast(Mock, result.group.size).return_value = 2
    calls: list[list[int]] = []

    def gather(row: mx.array, group: object) -> mx.array:
        assert group is result.group
        values = row.tolist()
        assert isinstance(values, list)
        integers: list[int] = []
        for value in values:
            assert isinstance(value, int)
            integers.append(value)
        calls.append(integers)
        return mx.concatenate([row, row])

    monkeypatch.setattr(mx.distributed, "all_gather", gather)
    assert not result._admit_memory(0, 65536, list(result._slots))
    assert 1 <= len(calls) <= 3 and all(row[0] == 0 for row in calls)
    assert all(row[1:4] == calls[0][1:4] for row in calls)


@pytest.mark.parametrize("terminal", [FinishedResponse(), CancelledResponse()])
def test_terminal_releases_active_cache_handle(
    monkeypatch: pytest.MonkeyPatch, terminal: FinishedResponse | CancelledResponse
) -> None:
    result = engine(monkeypatch)
    slot = result._slots[0]
    prefix = cast(_OwnerPrefixCache, slot.kv_prefix_cache)
    cache = cached()
    reference = weakref.ref(cache[0])
    prefix.active_cache = cache
    slot._active = active(CHAT_TASK)
    result._all_tasks[CHAT_TASK.task_id] = CHAT_TASK
    result._memory_plans[0] = 65536
    del cache

    def complete(
        current: _Slot,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        current._active = None
        return iter([(CHAT_TASK.task_id, terminal)])

    monkeypatch.setattr(_Slot, "step", complete)
    result._advance(0)
    gc.collect()
    assert prefix.active_cache is None and reference() is None
    assert result._memory_plans[0] == 0


def test_other_idle_eviction_preserves_chosen_warm_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = engine(monkeypatch)
    selected = cast(_OwnerPrefixCache, result._slots[0].kv_prefix_cache)
    other = cast(_OwnerPrefixCache, result._slots[1].kv_prefix_cache)
    warm = cached()
    selected.caches.append(warm)
    other.caches.append(cached())
    calls: list[int] = []

    def rows(chosen: int, requested: int, mask: int) -> list[list[int]]:
        calls.append(mask)
        ready = int(not other.caches)
        return [[ready, mask, chosen, requested, 0, 0, 0, 0, 0]]

    monkeypatch.setattr(result, "_memory_rows", rows)
    assert result._admit_memory(0, 65536, list(result._slots))
    assert selected.caches[0] is warm
    assert other.caches == []
    assert calls == [3, 3]


def test_active_decode_does_not_probe_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = engine(monkeypatch)
    result._slots[0]._active = active(CHAT_TASK)
    result._all_tasks[CHAT_TASK.task_id] = CHAT_TASK
    probes = Mock(side_effect=AssertionError("decode probed admission"))
    monkeypatch.setattr(result, "_memory_rows", probes)

    def no_progress(
        index: int,
    ) -> list[tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]]:
        return []

    monkeypatch.setattr(result, "_advance", no_progress)
    assert list(result.step()) == []
    probes.assert_not_called()


@pytest.mark.parametrize("busy", [False, True])
def test_unadmittable_work_errors_when_idle_or_waits_for_active_progress(
    monkeypatch: pytest.MonkeyPatch, busy: bool
) -> None:
    result = engine(monkeypatch)
    result._queue.append(CHAT_TASK)
    result._all_tasks[CHAT_TASK.task_id] = CHAT_TASK
    if busy:
        result._slots[1]._active = active(CHAT_TASK)

    def rendered(*args: object) -> str:
        return "rendered"

    monkeypatch.setattr(module.batch_generator, "apply_chat_template", rendered)

    def tokens(*args: object) -> list[int]:
        return [1, 2]

    monkeypatch.setattr(module, "encode_prompt", tokens)

    def unchanged_tokens(tokens: list[int], tokenizer: object) -> list[int]:
        return tokens

    monkeypatch.setattr(module, "fix_unmatched_think_end_tokens", unchanged_tokens)

    def denied(*args: object) -> bool:
        return False

    monkeypatch.setattr(result, "_admit_memory", denied)
    progress = Mock(return_value=[])
    errors = Mock()
    monkeypatch.setattr(result, "_advance", progress)
    monkeypatch.setattr(result, "_send_error", errors)
    emitted = list(result.step())
    if busy:
        assert list(result._queue) == [CHAT_TASK]
        progress.assert_called_once_with(1)
        errors.assert_not_called()
        assert emitted == []
    else:
        assert not result._queue
        errors.assert_called_once()
        assert isinstance(errors.call_args.args[1], MemoryError)
        assert len(emitted) == 1 and isinstance(emitted[0][1], FinishedResponse)


def test_close_releases_active_cache_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    result = engine(monkeypatch)
    prefix = cast(_OwnerPrefixCache, result._slots[0].kv_prefix_cache)
    cache = cached()
    reference = weakref.ref(cache[0])
    prefix.active_cache = cache
    del cache
    result.close()
    gc.collect()
    assert prefix.active_cache is None and reference() is None


def test_unknown_model_all_idle_emits_visible_resource_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = engine(monkeypatch)
    result.model = cast(Model, Mock())
    result._queue.append(CHAT_TASK)
    result._all_tasks[CHAT_TASK.task_id] = CHAT_TASK

    def rendered(*args: object) -> str:
        return "rendered"

    monkeypatch.setattr(module.batch_generator, "apply_chat_template", rendered)

    def tokens(*args: object) -> list[int]:
        return [1]

    monkeypatch.setattr(module, "encode_prompt", tokens)

    def unchanged_tokens(tokens: list[int], tokenizer: object) -> list[int]:
        return tokens

    monkeypatch.setattr(module, "fix_unmatched_think_end_tokens", unchanged_tokens)
    emitted = list(result.step())
    events = cast(Mock, result.event_sender.send).call_args_list
    assert len(events) == 1
    event = cast(object, events[0].args[0])
    assert isinstance(event, ChunkGenerated)
    chunk = event.chunk
    assert isinstance(chunk, ErrorChunk)
    assert chunk.finish_reason == "error"
    assert chunk.error_message == "cooperative_memory_admission_failed"
    assert not result._queue and len(emitted) == 1
    assert isinstance(emitted[0][1], FinishedResponse)
    assert emitted[0][1].task_status == TaskStatus.Failed


@pytest.mark.parametrize("bad", ["0", "typo"])
def test_bad_explicit_configuration_rejects_generator_startup(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    from dataclasses import replace

    result = engine(monkeypatch)
    monkeypatch.setenv("EXO_COOPERATIVE_CACHE_BUDGET_BYTES", bad)
    with pytest.raises(ValueError, match="invalid_cooperative_memory_configuration"):
        replace(result)


def test_divergent_rank_ownership_is_same_fatal_boundary_not_local_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranks = [engine(monkeypatch), engine(monkeypatch)]
    ranks[1]._slots[1]._active = active(CHAT_TASK)
    active_identity = ranks[1]._slots[1]._active

    def rendered(*args: object) -> str:
        return "rendered"

    monkeypatch.setattr(module.batch_generator, "apply_chat_template", rendered)

    def tokens(*args: object) -> list[int]:
        return [1]

    monkeypatch.setattr(module, "encode_prompt", tokens)

    def unchanged_tokens(tokens: list[int], tokenizer: object) -> list[int]:
        return tokens

    monkeypatch.setattr(module, "fix_unmatched_think_end_tokens", unchanged_tokens)
    gathered = [[0, 3, 0, 65536, 0, 0, 0, 0, 0], [0, 1, 0, 65536, 0, 0, 0, 0, 0]]
    failures: list[str] = []
    for result in ranks:
        result._queue.append(CHAT_TASK)
        result._all_tasks[CHAT_TASK.task_id] = CHAT_TASK

        def gathered_rows(*args: object) -> list[list[int]]:
            return gathered

        monkeypatch.setattr(result, "_memory_rows", gathered_rows)
        with pytest.raises(RuntimeError) as failure:
            list(result.step())
        failures.append(str(failure.value))
        assert list(result._queue) == [CHAT_TASK]
        cast(Mock, result.event_sender.send).assert_not_called()
    assert failures == ["cooperative_memory_owner_state_mismatch"] * 2
    assert ranks[1]._slots[1]._active is active_identity
