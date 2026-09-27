"""Two independent native cache owners with cooperative TP prefill/decode.

No batch padding, cache deepcopy, speculative batch approximation or additional
model residency. All ranks advance slots in the same agreed task order.
"""

import os
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import BinaryIO, cast

import mlx.core as mx

from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import GenerationTask, TaskId
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt
from exo.worker.engines.mlx.generator.generate import generation_stream
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.llm_inference import batch_generator
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator


@dataclass(eq=False)
class _Slot(SequentialGenerator):
    # Parent owns cross-rank agreement and distributes the agreed tasks. Child
    # callbacks still need cancellation agreement while a prefill is executing.
    parent: "CooperativeGenerator | None" = None

    def agree_on_tasks(self) -> None:
        if self.parent is not None:
            self.parent.agree_on_tasks()

    def agree_on_cancellations(self) -> None:
        if self.parent is not None:
            self.parent.agree_on_cancellations()


@dataclass(eq=False)
class CooperativeGenerator(SequentialGenerator):
    _slots: list[_Slot] = field(default_factory=list, init=False)
    _terminal_emitted: set[TaskId] = field(default_factory=set, init=False)
    _retired: deque[TaskId] = field(default_factory=deque, init=False)
    _retired_set: set[TaskId] = field(default_factory=set, init=False)
    _decoded: set[int] = field(default_factory=set, init=False)
    _advancing: set[int] = field(default_factory=set, init=False)
    _pending: list[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for index in range(2):
            slot = _Slot(
                model=self.model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=self.tool_parser,
                kv_prefix_cache=KVPrefixCache(self.group),
                model_id=self.model_id,
                device_rank=self.device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                parent=self,
                on_cooperative_prefill_progress=lambda i=index: self._interleave(i),
            )
            slot._cancelled_tasks = self._cancelled_tasks
            self._slots.append(slot)

    def submit(self, task: GenerationTask) -> None:
        if task.task_id in self._retired_set or task.task_id in self._all_tasks:
            return
        super().submit(task)

    def _retire(self, task_id: TaskId) -> None:
        self._all_tasks.pop(task_id, None)
        self._queue = deque(task for task in self._queue if task.task_id != task_id)
        self._maybe_queue = [
            task for task in self._maybe_queue if task.task_id != task_id
        ]
        self._cancelled_tasks.discard(task_id)
        self._terminal_emitted.discard(task_id)
        self._maybe_cancel = [
            task for task in self._maybe_cancel if task.task_id != task_id
        ]
        for slot in self._slots:
            slot._all_tasks.pop(task_id, None)
            slot._queue = deque(task for task in slot._queue if task.task_id != task_id)
        self._retired.append(task_id)
        self._retired_set.add(task_id)
        if len(self._retired) > 4096:
            self._retired_set.discard(self._retired.popleft())

    def warmup(self) -> None:
        super().warmup()
        for slot in self._slots:
            slot.check_for_cancel_every = self.check_for_cancel_every

    def _advance(
        self, index: int
    ) -> list[tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]]:
        self._advancing.add(index)
        try:
            slot = self._slots[index]
            if index not in self._decoded:
                slot.prefill_step_size_override = (
                    int(os.getenv("EXO_INTERLEAVED_PREFILL_STEP_SIZE", "256"))
                    if self._decoded
                    else None
                )
            result: list[
                tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
            ] = []
            for task_id, response in slot.step():
                if task_id not in self._all_tasks or task_id in self._terminal_emitted:
                    continue
                if self.should_cancel(task_id):
                    response = CancelledResponse()
                    if slot._active is not None and slot._active[0].task_id == task_id:
                        slot._active[1].close()
                        slot._active = None
                if isinstance(response, (CancelledResponse, FinishedResponse)):
                    self._terminal_emitted.add(task_id)
                result.append((task_id, response))
            if self._slots[index]._active is not None:
                self._decoded.add(index)
            else:
                self._decoded.discard(index)
            return result
        finally:
            self._advancing.discard(index)

    def _interleave(self, current: int) -> None:
        for index in sorted(self._decoded):
            if index == current or index in self._advancing:
                continue
            for task_id, response in self._advance(index):
                if isinstance(response, GenerationChunk):
                    if self.device_rank == 0:
                        task = self._all_tasks[task_id]
                        self.event_sender.send(
                            ChunkGenerated(command_id=task.command_id, chunk=response)
                        )
                else:
                    self._pending.append((task_id, response))

    def step(
        self,
    ) -> Iterator[
        tuple[TaskId, GenerationChunk | FinishedResponse | CancelledResponse]
    ]:
        self.agree_on_tasks()
        self.agree_on_cancellations()
        for index, slot in enumerate(self._slots):
            if slot._active is not None and self.should_cancel(slot._active[0].task_id):
                task_id = slot._active[0].task_id
                slot._active[1].close()
                slot._active = None
                self._decoded.discard(index)
                if task_id not in self._terminal_emitted:
                    self._terminal_emitted.add(task_id)
                    self._pending.append((task_id, CancelledResponse()))
            while (
                slot._active is None
                and not slot._queue
                and self._queue
                and self.should_cancel(self._queue[0].task_id)
            ):
                task_id = self._queue.popleft().task_id
                if task_id not in self._terminal_emitted:
                    self._terminal_emitted.add(task_id)
                    self._pending.append((task_id, CancelledResponse()))
        while self._queue:
            idle = [
                slot for slot in self._slots if slot._active is None and not slot._queue
            ]
            if not idle:
                break
            task = self._queue.popleft()
            prompt = batch_generator.apply_chat_template(
                self.tokenizer, task.task_params
            )
            tokens = fix_unmatched_think_end_tokens(
                encode_prompt(self.tokenizer, prompt), self.tokenizer
            )
            scores = [
                candidate.kv_prefix_cache.prefix_match_length(tokens)
                if candidate in idle and candidate.kv_prefix_cache is not None
                else -1
                for candidate in self._slots
            ]
            if self.group is not None:
                # Idle cache state can diverge after a local eviction. Never
                # choose a cache owner independently across TP ranks.
                metadata = [
                    candidate.kv_prefix_cache.affinity_metadata_digest()
                    if candidate in idle and candidate.kv_prefix_cache is not None
                    else 0
                    for candidate in self._slots
                ]
                gathered = mx.distributed.all_gather(
                    mx.array(scores + metadata), group=self.group
                )
                rows = cast(
                    list[list[int]],
                    gathered.reshape(self.group.size(), len(scores) * 2).tolist(),
                )
                if any(row != rows[0] for row in rows[1:]):
                    for candidate in idle:
                        if candidate.kv_prefix_cache is not None:
                            candidate.kv_prefix_cache.clear()
                    scores = [
                        0 if candidate in idle else -1 for candidate in self._slots
                    ]
            slot = self._slots[max(range(len(scores)), key=scores.__getitem__)]
            slot._all_tasks[task.task_id] = task
            slot._queue.append(task)
        result = self._pending
        self._pending = []
        for index, slot in enumerate(self._slots):
            if slot._active is not None or slot._queue:
                result.extend(self._advance(index))
        result.extend(self._pending)
        self._pending = []
        for task_id, response in result:
            if isinstance(response, (FinishedResponse, CancelledResponse)):
                self._retire(task_id)
        return iter(result)

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise ValueError(
            "Cooperative native slots do not support disaggregated prefill"
        )

    def close(self) -> None:
        for slot in self._slots:
            if slot._active is not None:
                slot._active[1].close()
                slot._active = None
            slot._queue.clear()
            slot._all_tasks.clear()
            if slot.kv_prefix_cache is not None:
                slot.kv_prefix_cache.clear()
            slot.on_cooperative_prefill_progress = None
            slot.parent = None
            slot.close()
        self._slots.clear()
        self._pending.clear()
        self._queue.clear()
        self._all_tasks.clear()
        self._decoded.clear()
        self._advancing.clear()
        if self.kv_prefix_cache is not None:
            self.kv_prefix_cache.clear()
        mx.synchronize(generation_stream)
        super().close()
        mx.clear_cache()
