import time
from typing import cast

import anyio
import pytest
from anyio.lowlevel import checkpoint

from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskTerminated,
)
from exo.shared.types.tasks import Task, TaskId, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import RunnerFailed, RunnerId
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import channel, mp_channel
from exo.worker.runner.bootstrap import RunnerTerminationError
from exo.worker.runner.supervisor import RunnerStdioHandler, RunnerSupervisor
from exo.worker.tests.unittests.conftest import get_bound_mlx_ring_instance


def _sleep_forever(*_args: object) -> None:
    time.sleep(1000)


class _DeadProcess:
    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2

    exitcode = -6

    def is_alive(self) -> bool:
        return False


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    command_id = CommandId("cmd-a")
    task = TextGeneration(
        task_id=TaskId("task-a"),
        instance_id=bound_instance.instance.instance_id,
        command_id=command_id,
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )
    supervisor.in_progress[task.task_id] = task
    supervisor.shutdown = lambda: None

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    terminal = await event_receiver.receive()
    assert isinstance(terminal, TaskTerminated)
    assert terminal.task_id == task.task_id
    assert terminal.runner_id == bound_instance.bound_runner_id
    assert not supervisor.in_progress

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_wait_stopped_resolves_only_after_process_actually_exits() -> None:
    """Regression test: main.py's Shutdown handling awaits wait_stopped()
    before the next plan() tick is allowed to create a replacement runner for
    the same instance. If wait_stopped() resolved before the OS process (and
    whatever resources it held, e.g. an RDMA queue pair) actually went away,
    a fast Shutdown->CreateRunner cycle could race the old process's
    teardown."""
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()

    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-a"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-a"),
        node_id=NodeId("node-a"),
    )

    runner_process = AsyncProcess(target=_sleep_forever, args=(), daemon=True)
    handler = await RunnerStdioHandler.create(
        stdout_rx=runner_process.stdout, stderr_rx=runner_process.stderr
    )
    supervisor = RunnerSupervisor(
        shard_metadata=bound_instance.bound_shard,
        bound_instance=bound_instance,
        runner_process=runner_process,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _event_sender=event_sender,
        _cancel_sender=cancel_sender,
    )

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor.run)

        with anyio.fail_after(5):
            while not runner_process.is_alive():
                await anyio.sleep(0.01)

        assert not supervisor._stopped.is_set()  # pyright: ignore[reportPrivateUsage]

        supervisor.shutdown()

        with anyio.fail_after(10):
            await supervisor.wait_stopped()

        assert not runner_process.is_alive()

        # Safe to await again once already stopped (level-triggered event).
        with anyio.fail_after(1):
            await supervisor.wait_stopped()

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@pytest.mark.anyio
async def test_cancel_queued_task_waits_for_runner_admission() -> None:
    """A cancel ID must not be consumed before Runner has submitted that task."""
    from exo.shared.types.events import TaskAcknowledged, TaskTerminated

    event_sender, event_receiver = channel[Event]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    runner_events, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound = get_bound_mlx_ring_instance(
        InstanceId("instance-a"),
        ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        RunnerId("runner-a"),
        NodeId("node-a"),
    )
    proc = cast(AsyncProcess, cast(object, _DeadProcess()))
    handler = await RunnerStdioHandler.create(
        stdout_rx=proc.stdout, stderr_rx=proc.stderr
    )
    supervisor = RunnerSupervisor(
        bound_instance=bound,
        shard_metadata=bound.bound_shard,
        runner_process=proc,
        _runner_stdio_handler=handler,
        initialize_timeout=400,
        _ev_recv=ev_recv,
        _task_sender=task_sender,
        _cancel_sender=cancel_sender,
        _event_sender=event_sender,
    )
    task = TextGeneration(
        instance_id=bound.instance.instance_id,
        command_id=CommandId("queued"),
        task_params=TextGenerationTaskParams(
            model=bound.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    cancelling = anyio.Event()
    cancelled = anyio.Event()

    async def cancel() -> None:
        cancelling.set()
        await supervisor.cancel_task(task.task_id)
        cancelled.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(supervisor.start_task, task)
        assert (await task_receiver.receive_async()).task_id == task.task_id
        tg.start_soon(cancel)
        await cancelling.wait()
        await checkpoint()
        assert not cancelled.is_set()
        assert not cancel_receiver.collect()
        runner_events.send(TaskAcknowledged(task_id=task.task_id))
        with anyio.fail_after(2):
            await cancelled.wait()
        assert cancel_receiver.collect() == [task.task_id]
        assert not event_receiver.collect()  # admission is not terminal ACK
        runner_events.send(
            TaskTerminated(task_id=task.task_id, runner_id=bound.bound_runner_id)
        )
        terminal = await event_receiver.receive()
        assert isinstance(terminal, TaskTerminated)
        assert task.task_id not in supervisor.in_progress
        # A task cancelled before dispatch must never subsequently enter Runner.
        never_started = task.model_copy(update={"task_id": TaskId("never-started")})
        await supervisor.cancel_task(never_started.task_id)
        assert isinstance(await event_receiver.receive(), TaskTerminated)
        await supervisor.start_task(never_started)
        assert not task_receiver.collect()
        assert never_started.task_id not in supervisor.pending
        tg.cancel_scope.cancel()
