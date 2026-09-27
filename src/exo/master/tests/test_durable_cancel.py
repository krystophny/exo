# pyright: reportPrivateUsage=false
"""Cancellation must stop the native engine, not merely close the API stream."""

import queue
from pathlib import Path
from typing import Callable, cast
from unittest.mock import Mock

import anyio
import pytest
from anyio import to_thread

import exo.master.main as master_module
from exo.api.main import API
from exo.master.main import Master
from exo.shared.types.chunks import ErrorChunk, TokenChunk
from exo.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from exo.shared.types.commands import TextGeneration as Generate
from exo.shared.types.common import CommandId, NodeId, SessionId, SystemId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    GlobalForwarderEvent,
    InstanceDeleted,
    LocalForwarderEvent,
    TaskDeleted,
    TaskStatusUpdated,
    TaskTerminated,
)
from exo.shared.types.tasks import CancelTask, Task, TaskId, TaskStatus, TextGeneration
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.shared.types.worker.runners import RunnerId, RunnerReady
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import MpSender, channel, mp_channel
from exo.worker.engines.base import Engine
from exo.worker.plan import _cancel_tasks
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.runner import Runner
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance
from exo.worker.tests.unittests.test_runner.test_event_ordering import CHAT_TASK


@pytest.mark.anyio
@pytest.mark.parametrize(
    "stop",
    ["explicit", "disconnect", "request_cancel", "normal", "failure", "guard_error"],
)
async def test_api_stop_reaches_engine_and_waits_all_original_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    monkeypatch.setattr(master_module, "EXO_EVENT_LOG_DIR", tmp_path)
    node = NodeId("node-a")
    rank = RunnerId("rank-a")
    sibling = RunnerId("rank-b")
    bound = get_bound_mlx_ring_instance(
        CHAT_TASK.instance_id, CHAT_TASK.task_params.model, rank, node
    )
    assignments = bound.instance.shard_assignments.model_copy(
        update={
            "runner_to_shard": {rank: bound.bound_shard, sibling: bound.bound_shard},
            "node_to_runner": {node: rank, NodeId("node-b"): sibling},
        }
    )
    instance = bound.instance.model_copy(update={"shard_assignments": assignments})
    bound = BoundInstance(instance=instance, bound_runner_id=rank, bound_node_id=node)
    command_sender, command_receiver = channel[ForwarderCommand]()
    event_sender, event_receiver = channel[Event]()
    local_sender, local_receiver = channel[LocalForwarderEvent]()
    global_sender, global_receiver = channel[GlobalForwarderEvent]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    session = SessionId(master_node_id=node, election_clock=0)
    master = Master(
        node,
        session,
        command_receiver=command_receiver,
        event_sender=event_sender,
        local_event_receiver=local_receiver,
        global_event_sender=global_sender,
        download_command_sender=download_sender,
    )
    master.state = master.state.model_copy(
        update={"instances": {instance.instance_id: instance}}
    )
    api = object.__new__(API)
    api.paused = False
    api.command_sender = command_sender
    api._system_id = SystemId("api")
    api._text_generation_queues = {}
    api._image_generation_queues = {}
    native: list[Event] = []
    indexed: list[Event] = []
    changed = anyio.Event()

    async def router() -> None:
        async for event in event_receiver:
            native.append(event)
            await local_sender.send(
                LocalForwarderEvent(
                    origin=SystemId("router"),
                    origin_idx=len(native) - 1,
                    session=session,
                    event=event,
                )
            )
            if isinstance(event, ChunkGenerated) and isinstance(
                event.chunk, (TokenChunk, ErrorChunk)
            ):
                sender = api._text_generation_queues.get(event.command_id)
                if sender is not None:
                    await sender.send(event.chunk)

    async def collect_indexed() -> None:
        nonlocal changed
        async for forwarded in global_receiver:
            indexed.append(forwarded.event)
            changed.set()
            changed = anyio.Event()

    async def until(predicate: Callable[[], bool]) -> None:
        with anyio.fail_after(3):
            while not predicate():
                await changed.wait()

    task_sender, _ = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    runner_sender, runner_receiver = mp_channel[Event | RunnerTerminationError]()
    process = Mock(spec=AsyncProcess)
    _, out = channel[bytes]()
    _, err = channel[bytes]()
    handler = await RunnerStdioHandler.create(stdout_rx=out, stderr_rx=err)
    supervisor = RunnerSupervisor(
        bound_instance=bound,
        shard_metadata=bound.bound_shard,
        runner_process=process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _task_sender=task_sender,
        _cancel_sender=cancel_sender,
        _ev_recv=runner_receiver,
        _event_sender=event_sender,
    )
    supervisor.status = RunnerReady()
    gates: queue.Queue[None] = queue.Queue()
    produced: dict[TaskId, list[int]] = {}
    tasks: dict[TaskId, TextGeneration] = {}
    cancelled: set[TaskId] = set()
    engine = Mock(spec=Engine)

    def submit(task: TextGeneration) -> None:
        tasks[task.task_id] = task
        produced[task.task_id] = []

    def step() -> list[
        tuple[TaskId, TokenChunk | CancelledResponse | FinishedResponse]
    ]:
        gates.get(timeout=3)
        cancelled.update(cancel_receiver.collect())
        results: list[
            tuple[TaskId, TokenChunk | CancelledResponse | FinishedResponse]
        ] = []
        for task_id in list(tasks):
            if stop == "guard_error" and task_id == victim_id:
                runner_sender.send(
                    ChunkGenerated(
                        command_id=command.command_id,
                        chunk=ErrorChunk(
                            model=CHAT_TASK.task_params.model,
                            error_message="resource budget exceeded",
                        ),
                    )
                )
                results.append(
                    (task_id, FinishedResponse(task_status=TaskStatus.Failed))
                )
                del tasks[task_id]
            elif task_id in cancelled:
                results.append((task_id, CancelledResponse()))
                del tasks[task_id]
            elif len(produced[task_id]) == 3:
                results.append((task_id, FinishedResponse()))
                del tasks[task_id]
            else:
                token = len(produced[task_id])
                produced[task_id].append(token)
                results.append(
                    (
                        task_id,
                        TokenChunk(
                            model=CHAT_TASK.task_params.model,
                            text=str(token),
                            token_id=token,
                            usage=None,
                            finish_reason="stop"
                            if token == 2 and stop == "normal"
                            else None,
                        ),
                    )
                )
        return results

    cast(Mock, engine.submit).side_effect = submit
    cast(Mock, engine.step).side_effect = step
    runner = Runner(bound, Mock(), cast(MpSender[Event], runner_sender), Mock())
    runner.generator = cast(Engine, engine)
    runner.current_status = RunnerReady()
    async with anyio.create_task_group() as tg:
        tg.start_soon(master._command_processor)
        tg.start_soon(master._event_processor)
        tg.start_soon(router)
        tg.start_soon(collect_indexed)
        tg.start_soon(supervisor._forward_events)
        command = Generate(task_params=CHAT_TASK.task_params)
        await api._send(command)
        await until(lambda: command.command_id in master.command_task_mapping)
        victim_id = master.command_task_mapping[command.command_id]
        await until(lambda: victim_id in master.state.tasks)
        victim = cast(TextGeneration, master.state.tasks[victim_id])
        survivor = CHAT_TASK.model_copy(
            update={
                "task_id": TaskId("survivor"),
                "command_id": CommandId("survivor-command"),
            }
        )
        runner.active_tasks[survivor.task_id] = survivor
        submit(survivor)
        supervisor.in_progress[victim_id] = victim
        supervisor.in_progress[survivor.task_id] = survivor
        stream = api._token_chunk_stream(command.command_id)
        scope = anyio.CancelScope()
        first = anyio.Event()
        stopped = anyio.Event()

        async def consume() -> None:
            if stop == "disconnect":
                await anext(stream)
                first.set()
                return
            with scope:
                async for _chunk in stream:
                    first.set()
            stopped.set()

        tg.start_soon(consume)
        tg.start_soon(to_thread.run_sync, runner.handle_generation_tasks, victim)
        gates.put(None)
        await first.wait()
        if stop in ("normal", "guard_error"):
            for _ in range(4):
                gates.put(None)
            await stopped.wait()
            if stop == "guard_error":
                await until(
                    lambda: any(
                        isinstance(e, TaskStatusUpdated)
                        and e.task_id == victim_id
                        and e.task_status == TaskStatus.Failed
                        for e in indexed
                    )
                )
                await until(
                    lambda: any(
                        isinstance(e, TaskTerminated) and e.task_id == victim_id
                        for e in indexed
                    )
                )
                assert master.state.tasks[victim_id].task_status == TaskStatus.Failed
                assert master.command_task_mapping[command.command_id] == victim_id
                assert not any(
                    isinstance(e, TaskDeleted) and e.task_id == victim_id
                    for e in indexed
                )
                await event_sender.send(
                    TaskTerminated(task_id=victim_id, runner_id=sibling)
                )
            await until(
                lambda: any(
                    isinstance(e, TaskDeleted) and e.task_id == victim_id
                    for e in indexed
                )
            )
            await until(
                lambda: any(
                    isinstance(e, TaskTerminated) and e.task_id == survivor.task_id
                    for e in indexed
                )
            )
            assert produced[victim_id] == ([] if stop == "guard_error" else [0, 1, 2])
            if stop == "guard_error":
                assert any(
                    isinstance(e, ChunkGenerated)
                    and e.command_id == command.command_id
                    and isinstance(e.chunk, ErrorChunk)
                    for e in indexed
                )
                assert any(
                    isinstance(e, TaskStatusUpdated)
                    and e.task_id == victim_id
                    and e.task_status == TaskStatus.Failed
                    for e in indexed
                )
                assert not any(
                    isinstance(e, TaskStatusUpdated)
                    and e.task_id == victim_id
                    and e.task_status == TaskStatus.Complete
                    for e in indexed
                )
                assert (
                    sum(
                        isinstance(e, TaskTerminated) and e.task_id == victim_id
                        for e in indexed
                    )
                    == 2
                )
            assert not any(
                isinstance(e, TaskStatusUpdated)
                and e.task_id == victim_id
                and e.task_status == TaskStatus.Cancelled
                for e in indexed
            )
            assert command.command_id not in master.command_task_mapping
            tg.cancel_scope.cancel()
            return
        if stop == "explicit":
            await api.cancel_command(command.command_id)
        elif stop in ("request_cancel", "failure"):
            api.paused = True
            scope.cancel()
        else:
            api.paused = True
            await stream.aclose()
            stopped.set()
        await stopped.wait()
        await until(
            lambda: master.state.tasks[victim_id].task_status == TaskStatus.Cancelled
        )
        # Frontend finally has run. Native cancellation remains visible to planner.
        assert master.command_task_mapping[command.command_id] == victim_id
        if stop == "failure":
            await event_sender.send(InstanceDeleted(instance_id=instance.instance_id))
            await until(
                lambda: any(
                    isinstance(e, TaskDeleted) and e.task_id == victim_id
                    for e in indexed
                )
            )
            errors = [
                e.chunk
                for e in indexed
                if isinstance(e, ChunkGenerated)
                and e.command_id == command.command_id
                and isinstance(e.chunk, ErrorChunk)
            ]
            assert len(errors) == 1
            assert (
                "unconfirmed runners: ['rank-a', 'rank-b']" in errors[0].error_message
            )
            assert not any(
                isinstance(e, TaskTerminated) and e.task_id == victim_id
                for e in indexed
            )
            assert instance.instance_id not in master.state.instances
            assert command.command_id not in master.command_task_mapping
            assert victim_id not in master._task_runners
            # Failure-abandon makes no engine-stop claim. Deliberately let the old
            # engine continue, proving late output/Complete cannot be delivered.
            for _ in range(4):
                gates.put(None)
            await until(
                lambda: any(
                    isinstance(e, TaskTerminated) and e.task_id == survivor.task_id
                    for e in indexed
                )
            )
            assert produced[victim_id] == [0, 1, 2]
            delivered = [
                e.chunk.token_id
                for e in indexed
                if isinstance(e, ChunkGenerated)
                and e.command_id == command.command_id
                and isinstance(e.chunk, TokenChunk)
            ]
            assert delivered == [0]
            assert not any(
                isinstance(e, TaskStatusUpdated)
                and e.task_id == victim_id
                and e.task_status == TaskStatus.Complete
                for e in indexed
            )
            assert victim_id not in master.state.tasks
            assert victim_id not in master._terminated_runners
            tg.cancel_scope.cancel()
            return
        plan = _cancel_tasks({rank: supervisor}, master.state.tasks)
        assert isinstance(plan, CancelTask) and plan.cancelled_task_id == victim_id
        await supervisor.cancel_task(victim_id)
        for _ in range(4):
            gates.put(None)
        await until(
            lambda: any(
                isinstance(e, TaskTerminated) and e.task_id == victim_id
                for e in indexed
            )
        )
        await until(
            lambda: any(
                isinstance(e, TaskTerminated) and e.task_id == survivor.task_id
                for e in indexed
            )
        )
        assert produced[victim_id] == [0]
        assert produced[survivor.task_id] == [0, 1, 2]
        assert victim_id in master.state.tasks  # sibling ACK still required
        assert master.state.tasks[victim_id].task_status == TaskStatus.Cancelled
        assert not any(
            isinstance(e, TaskStatusUpdated)
            and e.task_id == victim_id
            and e.task_status == TaskStatus.Complete
            for e in indexed
        )
        # Replacement runner is not an original owner; duplicate ACK cannot retire.
        await event_sender.send(
            TaskTerminated(task_id=victim_id, runner_id=RunnerId("replacement"))
        )
        await event_sender.send(TaskTerminated(task_id=victim_id, runner_id=rank))
        await event_sender.send(
            TaskStatusUpdated(task_id=victim_id, task_status=TaskStatus.Complete)
        )
        assert victim_id in master.state.tasks
        await event_sender.send(TaskTerminated(task_id=victim_id, runner_id=sibling))
        await until(
            lambda: any(
                isinstance(e, TaskDeleted) and e.task_id == victim_id for e in indexed
            )
        )
        assert victim_id not in master.state.tasks
        await event_sender.send(
            TaskStatusUpdated(task_id=victim_id, task_status=TaskStatus.Complete)
        )
        late_ack = TaskTerminated(task_id=victim_id, runner_id=sibling)
        await event_sender.send(late_ack)
        await until(lambda: any(e.event_id == late_ack.event_id for e in indexed))
        assert command.command_id not in master.command_task_mapping
        assert victim_id not in master._task_runners
        assert victim_id not in master._terminated_runners
        assert not any(
            isinstance(e, TaskStatusUpdated)
            and e.task_id == victim_id
            and e.task_status == TaskStatus.Complete
            for e in indexed
        )
        assert (
            sum(isinstance(e, TaskDeleted) and e.task_id == victim_id for e in indexed)
            == 1
        )
        tg.cancel_scope.cancel()
