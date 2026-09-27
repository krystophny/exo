# pyright: reportPrivateUsage=false
"""A resource rejection must retire the engine task without reporting success."""

from typing import cast
from unittest.mock import Mock

from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    TaskStatusUpdated,
    TaskTerminated,
)
from exo.shared.types.tasks import TaskId, TaskStatus
from exo.shared.types.worker.runner_response import FinishedResponse
from exo.shared.types.worker.runners import RunnerReady
from exo.utils.channels import MpSender
from exo.worker.engines.base import Engine
from exo.worker.runner.runner import ExitCode, Runner
from exo.worker.tests.constants import INSTANCE_1_ID, MODEL_A_ID, NODE_A, RUNNER_1_ID
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance
from exo.worker.tests.unittests.test_runner.test_event_ordering import CHAT_TASK


def test_runner_resource_error_has_failed_terminal_and_exactly_one_ack() -> None:
    events: list[Event] = []
    sender = Mock()
    cast(Mock, sender.send).side_effect = events.append
    engine = Mock(spec=Engine)
    native_steps = 0

    def reject() -> list[tuple[TaskId, FinishedResponse]]:
        nonlocal native_steps
        native_steps += 1
        cast(Mock, sender.send)(
            ChunkGenerated(
                command_id=CHAT_TASK.command_id,
                chunk=ErrorChunk(
                    model=MODEL_A_ID, error_message="resource budget exceeded"
                ),
            )
        )
        return [(CHAT_TASK.task_id, FinishedResponse(task_status=TaskStatus.Failed))]

    cast(Mock, engine.step).side_effect = reject
    bound = get_bound_mlx_ring_instance(INSTANCE_1_ID, MODEL_A_ID, RUNNER_1_ID, NODE_A)
    runner = Runner(bound, Mock(), cast(MpSender[Event], sender), Mock())
    runner.generator = cast(Engine, engine)
    runner.current_status = RunnerReady()
    assert runner.handle_generation_tasks(CHAT_TASK) == ExitCode.AllTasksComplete
    errors = [e for e in events if isinstance(e, ChunkGenerated)]
    assert len(errors) == 1
    assert isinstance(errors[0].chunk, ErrorChunk)
    assert errors[0].chunk.finish_reason == "error"
    statuses = [
        e.task_status
        for e in events
        if isinstance(e, TaskStatusUpdated) and e.task_id == CHAT_TASK.task_id
    ]
    assert statuses == [TaskStatus.Failed]
    acks = [e for e in events if isinstance(e, TaskTerminated)]
    assert len(acks) == 1
    assert acks[0].task_id == CHAT_TASK.task_id and acks[0].runner_id == RUNNER_1_ID
    assert (
        events.index(errors[0])
        < next(
            i
            for i, e in enumerate(events)
            if isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Failed
        )
        < events.index(acks[0])
    )
    assert native_steps == 1 and not runner.active_tasks
