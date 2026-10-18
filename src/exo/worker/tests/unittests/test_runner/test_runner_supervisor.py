import time
from typing import cast, final

import anyio
import pytest
from anyio.lowlevel import checkpoint

import exo.worker.runner.supervisor as supervisor_module
from exo.shared.models.model_cards import ModelId
from exo.shared.types.chunks import ErrorChunk
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
    TaskTerminated,
)
from exo.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus, TextGeneration
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.instances import BoundInstance, InstanceId
from exo.shared.types.worker.runners import (
    RunnerFailed,
    RunnerId,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
)
from exo.utils.async_process import AsyncProcess
from exo.utils.channels import Receiver, Sender, channel, mp_channel
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


async def _supervisor_of_dead_runner(event_sender: Sender[Event]) -> RunnerSupervisor:
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
    supervisor.shutdown = lambda: None
    return supervisor


def _text_generation(supervisor: RunnerSupervisor, name: str) -> TextGeneration:
    return TextGeneration(
        task_id=TaskId(f"task-{name}"),
        instance_id=supervisor.bound_instance.instance.instance_id,
        command_id=CommandId(f"cmd-{name}"),
        task_params=TextGenerationTaskParams(
            model=supervisor.bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
            stream=True,
        ),
    )


@pytest.mark.anyio
async def test_check_runner_emits_error_chunk_for_inflight_text_generation() -> None:
    event_sender, event_receiver = channel[Event]()
    supervisor = await _supervisor_of_dead_runner(event_sender)
    task = _text_generation(supervisor, "a")
    supervisor.in_progress[task.task_id] = task

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    got_chunk = await event_receiver.receive()
    got_status = await event_receiver.receive()

    assert isinstance(got_chunk, ChunkGenerated)
    assert got_chunk.command_id == task.command_id
    assert isinstance(got_chunk.chunk, ErrorChunk)
    assert "Runner shutdown before completing command" in got_chunk.chunk.error_message

    assert isinstance(got_status, RunnerStatusUpdated)
    assert isinstance(got_status.runner_status, RunnerFailed)

    terminal = await event_receiver.receive()
    assert isinstance(terminal, TaskTerminated)
    assert terminal.task_id == task.task_id
    assert terminal.runner_id == supervisor.bound_instance.bound_runner_id
    assert not supervisor.in_progress

    event_sender.close()
    with anyio.move_on_after(0.1):
        await event_receiver.aclose()


@final
class _FinishesATaskWhileSending:
    """An event sender during whose sends the runner's last results arrive.

    While the supervisor reports one request as failed, the event loop runs the coroutine that
    forwards the runner's output, which marks another request complete.
    """

    def __init__(self) -> None:
        self.sent: list[Event] = []
        self.supervisor: RunnerSupervisor | None = None

    async def send(self, event: Event) -> None:
        self.sent.append(event)
        assert self.supervisor is not None
        if len(self.supervisor.in_progress) > 1:
            self.supervisor.in_progress.pop(next(reversed(self.supervisor.in_progress)))
        await anyio.sleep(0)


@pytest.mark.anyio
async def test_check_runner_survives_a_request_finishing_while_it_reports() -> None:
    sender = _FinishesATaskWhileSending()
    supervisor = await _supervisor_of_dead_runner(
        cast(Sender[Event], cast(object, sender))
    )
    sender.supervisor = supervisor
    first, second = _text_generation(supervisor, "a"), _text_generation(supervisor, "b")
    supervisor.in_progress[first.task_id] = first
    supervisor.in_progress[second.task_id] = second

    await supervisor._check_runner(RuntimeError("boom"))  # pyright: ignore[reportPrivateUsage]

    failed = [e.command_id for e in sender.sent if isinstance(e, ChunkGenerated)]
    assert failed == [first.command_id]
    # The failure status is followed only by the terminal report of the request
    # that was still in flight once the runner process had gone.
    status_index = next(
        i for i, e in enumerate(sender.sent) if isinstance(e, RunnerStatusUpdated)
    )
    after = sender.sent[status_index + 1 :]
    assert [e.task_id for e in after if isinstance(e, TaskTerminated)] == [
        first.task_id
    ]
    assert all(isinstance(e, TaskTerminated) for e in after)


class _StuckProcess:
    """A runner process that is alive but will never make progress again."""

    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def stop(self) -> None:
        self.exitcode = -15


@pytest.mark.anyio
async def test_a_runner_whose_ring_aborted_is_stopped_and_reported_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-b"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-b"),
        node_id=NodeId("node-b"),
    )
    stuck = _StuckProcess()
    proc = cast(AsyncProcess, cast(object, stuck))
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
    supervisor.shutdown = lambda: None
    for _ in range(10):
        handler.diagnostics.record_line(
            "[ring] Receiving from socket 51 failed with errno 54"
        )
    handler.diagnostics.record_line("[ring] Too many send/recv errors. Aborting...")

    with anyio.fail_after(2):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        status = await event_receiver.receive()

    assert not stuck.is_alive()
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)


@pytest.mark.anyio
async def test_a_healthy_runner_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    event_sender, _ = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-c"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-c"),
        node_id=NodeId("node-c"),
    )
    healthy = _StuckProcess()
    proc = cast(AsyncProcess, cast(object, healthy))
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
    handler.diagnostics.record_line("some unrelated warning")

    with anyio.move_on_after(0.1):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]

    assert healthy.is_alive()


class _AliveProcess:
    """A runner process that stays alive until it is stopped."""

    def __init__(self):
        rx1, _ = channel[bytes]()
        rx2, _ = channel[bytes]()
        self.stdout = rx1
        self.stderr = rx2
        self.exitcode: int | None = None

    def is_alive(self) -> bool:
        return self.exitcode is None

    async def stop(self) -> None:
        self.exitcode = -15


async def generating_supervisor(
    rank: int,
) -> tuple[RunnerSupervisor, _AliveProcess, Receiver[Event]]:
    event_sender, event_receiver = channel[Event]()
    task_sender, _ = mp_channel[Task]()
    cancel_sender, _ = mp_channel[TaskId]()
    _, ev_recv = mp_channel[Event | RunnerTerminationError]()
    bound_instance: BoundInstance = get_bound_mlx_ring_instance(
        instance_id=InstanceId("instance-s"),
        model_id=ModelId("mlx-community/Llama-3.2-1B-Instruct-4bit"),
        runner_id=RunnerId("runner-s"),
        node_id=NodeId("node-s"),
    )
    if rank != 0:
        bound_instance = BoundInstance(
            instance=bound_instance.instance,
            bound_runner_id=RunnerId("other_runner"),
            bound_node_id=NodeId("other_node"),
        )
    process = _AliveProcess()
    proc = cast(AsyncProcess, cast(object, process))
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
    supervisor.shutdown = lambda: None
    supervisor.status = RunnerRunning()
    task = TextGeneration(
        task_id=TaskId("task-s"),
        instance_id=bound_instance.instance.instance_id,
        command_id=CommandId("cmd-s"),
        task_params=TextGenerationTaskParams(
            model=bound_instance.bound_shard.model_card.model_id,
            input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        ),
    )
    supervisor.in_progress[task.task_id] = task
    return supervisor, process, event_receiver


@pytest.mark.anyio
async def test_a_first_rank_runner_silent_mid_generation_is_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, events = await generating_supervisor(rank=0)

    with anyio.fail_after(2):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        chunk = await events.receive()
        status = await events.receive()

    assert not process.is_alive()
    assert isinstance(chunk, ChunkGenerated)
    assert isinstance(chunk.chunk, ErrorChunk)
    assert isinstance(status, RunnerStatusUpdated)
    assert isinstance(status.runner_status, RunnerFailed)


@pytest.mark.anyio
async def test_a_runner_that_keeps_reporting_is_left_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The runner reports every 10 ms; the stall timeout leaves room for a busy CI machine to pause the event loop
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.5)
    supervisor, process, _ = await generating_supervisor(rank=0)

    async def keep_reporting() -> None:
        while True:
            supervisor._last_heard = anyio.current_time()  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep(0.01)

    async with anyio.create_task_group() as tg:
        tg.start_soon(keep_reporting)
        with anyio.move_on_after(1.0):
            await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]
        tg.cancel_scope.cancel()

    assert process.is_alive()


@pytest.mark.anyio
async def test_other_ranks_are_not_judged_by_their_silence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor_module, "RUNNER_WATCH_INTERVAL", 0.01)
    monkeypatch.setattr(supervisor_module, "RUNNER_STALL_TIMEOUT", 0.05)
    supervisor, process, _ = await generating_supervisor(rank=1)

    with anyio.move_on_after(0.3):
        await supervisor._watch_runner()  # pyright: ignore[reportPrivateUsage]

    assert process.is_alive()


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

    async def hold_supervisor_task_group() -> None:
        async with supervisor._tg:  # pyright: ignore[reportPrivateUsage]
            await anyio.sleep_forever()

    async with anyio.create_task_group() as tg:
        # cancel_task defers the send to the supervisor's own task group.
        tg.start_soon(hold_supervisor_task_group)
        with anyio.fail_after(2):
            while not supervisor._tg.is_running():  # pyright: ignore[reportPrivateUsage]
                await checkpoint()
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        tg.start_soon(supervisor.start_task, task)
        assert (await task_receiver.receive_async()).task_id == task.task_id
        tg.start_soon(cancel)
        await cancelling.wait()
        # The caller is not held up while the runner has not admitted the task...
        with anyio.fail_after(2):
            await cancelled.wait()
        await checkpoint()
        # ...but the cancel ID is not sent (and so cannot be lost) before admission.
        assert not cancel_receiver.collect()
        runner_events.send(TaskAcknowledged(task_id=task.task_id))
        with anyio.fail_after(2):
            while not (sent := cancel_receiver.collect()):
                await anyio.sleep(0.01)
        assert sent == [task.task_id]
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


@pytest.mark.anyio
async def test_shutdown_waits_for_the_runners_final_events() -> None:
    """A Shutdown is acknowledged before the runner releases its model; the
    supervisor reports shut-down only after forwarding the runner's last events."""
    event_sender, event_receiver = channel[Event]()
    task_sender, _task_receiver = mp_channel[Task]()
    cancel_sender, _cancel_receiver = mp_channel[TaskId]()
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
    shutdown = Shutdown(
        instance_id=bound.instance.instance_id, runner_id=bound.bound_runner_id
    )
    async with anyio.create_task_group() as tg:
        tg.start_soon(supervisor._forward_events)  # pyright: ignore[reportPrivateUsage]
        runner_events.send(
            RunnerStatusUpdated(
                runner_id=bound.bound_runner_id, runner_status=RunnerShuttingDown()
            )
        )
        runner_events.send(TaskAcknowledged(task_id=shutdown.task_id))
        with anyio.move_on_after(0.2) as early:
            await supervisor.wait_runner_shut_down()
        assert early.cancelled_caught
        runner_events.send(
            TaskStatusUpdated(task_id=shutdown.task_id, task_status=TaskStatus.Complete)
        )
        runner_events.send(
            RunnerStatusUpdated(
                runner_id=bound.bound_runner_id, runner_status=RunnerShutdown()
            )
        )
        with anyio.fail_after(2):
            await supervisor.wait_runner_shut_down()
        forwarded = event_receiver.collect()
        assert any(
            isinstance(e, TaskStatusUpdated) and e.task_status == TaskStatus.Complete
            for e in forwarded
        )
        assert isinstance(forwarded[-1], RunnerStatusUpdated)
        assert isinstance(forwarded[-1].runner_status, RunnerShutdown)
        tg.cancel_scope.cancel()
