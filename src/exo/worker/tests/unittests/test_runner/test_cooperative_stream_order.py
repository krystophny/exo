# pyright: reportPrivateUsage=false
"""Actual Runner delivery must preserve each request's native token order."""

from collections.abc import Generator, Iterator
from typing import cast
from unittest.mock import Mock

import pytest

from exo.shared.types.chunks import GenerationChunk, TokenChunk
from exo.shared.types.events import ChunkGenerated, Event, TaskStatusUpdated
from exo.shared.types.tasks import TaskId, TaskStatus, TextGeneration
from exo.shared.types.worker.runner_response import (
    CancelledResponse,
    FinishedResponse,
    GenerationResponse,
)
from exo.shared.types.worker.runners import RunnerReady
from exo.utils.channels import MpSender
from exo.worker.runner.llm_inference.batch_generator import GeneratorQueue
from exo.worker.runner.llm_inference.cooperative_generator import (
    CooperativeGenerator,
    _Slot,
)
from exo.worker.runner.runner import ExitCode, Runner

from ...constants import COMMAND_1_ID, INSTANCE_1_ID, MODEL_A_ID, NODE_A, RUNNER_1_ID
from ..conftest import get_bound_mlx_ring_instance
from .test_event_ordering import CHAT_TASK


def active(
    task: TextGeneration,
) -> tuple[
    TextGeneration,
    Generator[GenerationResponse],
    GeneratorQueue[GenerationResponse],
    Iterator[GenerationChunk | None],
]:
    def stream() -> Generator[GenerationResponse]:
        while True:
            yield Mock()

    return task, stream(), GeneratorQueue(), iter([])


@pytest.mark.parametrize("terminal", [FinishedResponse(), CancelledResponse()])
@pytest.mark.parametrize("prefill_boundaries", [3, 4])
def test_runner_delivers_prior_decode_before_competing_prefill_interleaves(
    monkeypatch: pytest.MonkeyPatch,
    terminal: FinishedResponse | CancelledResponse,
    prefill_boundaries: int,
) -> None:
    events: list[Event] = []
    sender = Mock()
    cast(Mock, sender.send).side_effect = events.append
    engine = CooperativeGenerator(
        model=Mock(),
        tokenizer=Mock(),
        group=None,
        kv_prefix_cache=None,
        tool_parser=None,
        model_id=MODEL_A_ID,
        device_rank=0,
        cancel_receiver=Mock(),
        event_sender=cast(MpSender[Event], sender),
    )
    monkeypatch.setattr(engine, "agree_on_tasks", Mock())
    monkeypatch.setattr(engine, "agree_on_cancellations", Mock())
    first = CHAT_TASK
    second = CHAT_TASK.model_copy(
        update={
            "task_id": TaskId("00000000-0000-0000-0000-000000000002"),
            "command_id": "00000000-0000-0000-0000-000000000003",
        }
    )
    engine._all_tasks.update({first.task_id: first, second.task_id: second})
    engine._slots[0]._active = active(first)
    engine._slots[1]._queue.append(second)
    engine._slots[1]._all_tasks[second.task_id] = second
    first_tokens = iter(enumerate(["The", " user", " wants", " me"]))
    started_second = False

    def slot_step(
        slot: _Slot,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        nonlocal started_second
        if slot is engine._slots[0]:
            item = next(first_tokens, None)
            if item is None:
                slot._active = None
                return iter([(first.task_id, terminal)])
            token_id, text = item
            return iter(
                [
                    (
                        first.task_id,
                        TokenChunk(
                            model=MODEL_A_ID,
                            text=text,
                            token_id=token_id,
                            usage=None,
                        ),
                    )
                ]
            )
        if not started_second:
            started_second = True
            slot._queue.clear()
            slot._active = active(second)
            # Three prefill boundaries advance the existing decode owner. These
            # use its real _advance/_interleave path and the same sender as Runner.
            for _ in range(prefill_boundaries):
                engine._interleave(1)
            return iter(
                [
                    (
                        second.task_id,
                        TokenChunk(
                            model=MODEL_A_ID,
                            text="other",
                            token_id=10,
                            usage=None,
                        ),
                    )
                ]
            )
        slot._active = None
        return iter([(second.task_id, FinishedResponse())])

    monkeypatch.setattr(_Slot, "step", slot_step)
    runner = Runner(
        get_bound_mlx_ring_instance(INSTANCE_1_ID, MODEL_A_ID, RUNNER_1_ID, NODE_A),
        Mock(),
        cast(MpSender[Event], sender),
        Mock(),
    )
    runner.generator = engine
    runner.current_status = RunnerReady()
    runner.active_tasks[second.task_id] = second
    assert runner.handle_generation_tasks(first) == ExitCode.AllTasksComplete
    delivered = [
        e.chunk
        for e in events
        if isinstance(e, ChunkGenerated)
        and e.command_id == COMMAND_1_ID
        and isinstance(e.chunk, TokenChunk)
    ]
    # The independent backend script defines order. No sorting, splicing or
    # reconstruction of events may repair the actual delivery sequence.
    assert [c.token_id for c in delivered] == [0, 1, 2, 3]
    assert "".join(c.text for c in delivered) == "The user wants me"
    statuses = [
        e.task_status
        for e in events
        if isinstance(e, TaskStatusUpdated) and e.task_id == first.task_id
    ]
    assert statuses == (
        [TaskStatus.Complete] if isinstance(terminal, FinishedResponse) else []
    )
    assert not runner.active_tasks and not engine._all_tasks
    assert list(engine._retired).count(first.task_id) == 1
    assert list(engine._retired).count(second.task_id) == 1
