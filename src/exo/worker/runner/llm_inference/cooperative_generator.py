"""Two independent native cache owners with cooperative TP prefill/decode.

No batch padding, cache deepcopy, speculative batch approximation or additional
model residency. All ranks advance slots in the same agreed task order.
"""

import os
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, BinaryIO, Protocol, cast

import mlx.core as mx
from mlx_lm.models.cache import KVCache, MLACacheList, QuantizedKVCache
from mlx_lm.models.deepseek_v32 import Model as GlmModel

from exo.shared.types.chunks import GenerationChunk
from exo.shared.types.events import ChunkGenerated
from exo.shared.types.tasks import GenerationTask, TaskId, TaskStatus
from exo.shared.types.worker.runner_response import CancelledResponse, FinishedResponse
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import KVPrefixCache, encode_prompt
from exo.worker.engines.mlx.generator.generate import generation_stream
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.llm_inference import batch_generator
from exo.worker.runner.llm_inference.batch_generator import SequentialGenerator
from exo.worker.runner.llm_inference.cooperative_memory import (
    MemoryLimits,
    PhysicalSnapshot,
    capture_physical,
    projected_cache,
    reservation_bytes,
)

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import MediaRegion


class _CacheMembers(Protocol):
    caches: Sequence[object]


class _QuantizedStorage(Protocol):
    bits: int
    group_size: int


class _OwnerPrefixCache(KVPrefixCache):
    active_cache: KVCacheType | None = None

    def get_kv_cache(
        self,
        model: Model,
        prompt_tokens: mx.array,
        media_regions: list["MediaRegion"] | None = None,
    ) -> tuple[KVCacheType, mx.array, int | None, bool]:
        result = super().get_kv_cache(model, prompt_tokens, media_regions)
        self.active_cache = result[0]
        return result

    def release_active(self) -> None:
        self.active_cache = None

    def clear(self) -> None:
        super().clear()
        self.release_active()


@dataclass(eq=False)
class _Slot(SequentialGenerator):
    # Parent owns cross-rank agreement and distributes the agreed tasks. Child
    # callbacks still need cancellation agreement while a prefill is executing.
    parent: "CooperativeGenerator | None" = None

    def release_cache_handle(self) -> None:
        if isinstance(self.kv_prefix_cache, _OwnerPrefixCache):
            self.kv_prefix_cache.release_active()

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

    _memory_limits: MemoryLimits | None = field(default=None, init=False)
    _memory_baseline: PhysicalSnapshot = field(
        default_factory=PhysicalSnapshot, init=False
    )
    _memory_model_identity: int = field(default=0, init=False)
    _memory_plans: list[int] = field(default_factory=lambda: [0, 0], init=False)

    def __post_init__(self) -> None:
        self._memory_limits = MemoryLimits.configured()
        self._memory_model_identity = id(self.model)
        if self._memory_limits is not None:
            self._memory_baseline = capture_physical()
        for index in range(2):
            slot = _Slot(
                model=self.model,
                tokenizer=self.tokenizer,
                group=self.group,
                tool_parser=self.tool_parser,
                kv_prefix_cache=_OwnerPrefixCache(self.group),
                model_id=self.model_id,
                device_rank=self.device_rank,
                cancel_receiver=self.cancel_receiver,
                event_sender=self.event_sender,
                parent=self,
                on_cooperative_prefill_progress=lambda i=index: self._interleave(i),
            )
            slot._cancelled_tasks = self._cancelled_tasks
            self._slots.append(slot)

    def _owner_buffers(self) -> tuple[list[int], bool]:
        trees: list[list[object]] = []
        for slot in self._slots:
            prefix = slot.kv_prefix_cache
            if not isinstance(prefix, _OwnerPrefixCache):
                raise ValueError("unsupported_native_cache_owner")
            owned = list(prefix.caches)
            if prefix.active_cache is not None:
                owned.append(prefix.active_cache)
            states: list[object] = []
            for cache in owned:
                for entry in cache:
                    if not isinstance(entry, MLACacheList):
                        raise ValueError("unsupported_glm_cache_layout")
                    members = cast(_CacheMembers, cast(object, entry)).caches
                    if len(members) not in (
                        1,
                        2,
                    ):
                        raise ValueError("unsupported_glm_cache_layout")
                    for index, member in enumerate(members):
                        if index == 0:
                            if not isinstance(member, (KVCache, QuantizedKVCache)):
                                raise ValueError("unsupported_glm_latent_cache")
                            if isinstance(member, QuantizedKVCache):
                                storage = cast(_QuantizedStorage, cast(object, member))
                                if storage.bits != 8 or storage.group_size != 64:
                                    raise ValueError("unsupported_glm_quantization")
                            elif member.keys is not None:
                                raise ValueError("unquantized_glm_latent_cache")
                        elif type(member) is not KVCache:
                            raise ValueError("unsupported_glm_index_cache")
                        states.append(cast(object, member.state))
            trees.append(states)
        # Unlike .nbytes, the known MLX fork API measures full backing buffers,
        # including views, once per allocator allocation. Unevaluated trees fail.
        mx.eval(cast("mx.MX_ARRAY_TREE", trees))
        measure = cast(Callable[[object], int], vars(mx)["get_array_buffer_size"])
        allocated = [measure(tree) for tree in trees]
        nonoverlap = sum(allocated) == measure(trees)
        return allocated, nonoverlap

    def _memory_row(self, chosen: int, requested: int, idle_mask: int) -> list[int]:
        limits = self._memory_limits
        if limits is None:
            return [0, idle_mask, chosen, requested, 0, 0, 0, 0, 0]
        try:
            allocated, nonoverlap = self._owner_buffers()
            projected, growth = projected_cache(
                self._memory_plans, allocated, chosen, requested
            )
            physical = capture_physical()
            args = cast(GlmModel, cast(object, self.model)).args
            known_model = isinstance(self.model, GlmModel) and (
                args.num_hidden_layers == 78
                and args.kv_lora_rank == 512
                and args.qk_rope_head_dim == 64
                and args.index_head_dim == 128
                and args.indexer_types is not None
                and len(cast(list[str], args.indexer_types)) == 78
                and cast(list[str], args.indexer_types).count("full") == 21
            )
            ready = (
                requested > 0
                and nonoverlap
                and known_model
                and id(self.model) == self._memory_model_identity
                and os.getenv("EXO_KV_BITS") == "8"
                and physical.healthy(self._memory_baseline, time.monotonic())
                and projected <= limits.cache_bytes
                and growth + limits.physical_margin_bytes <= physical.headroom
            )
            return [
                int(ready),
                idle_mask,
                chosen,
                requested,
                projected,
                growth,
                physical.headroom,
                mx.get_active_memory(),
                mx.get_cache_memory(),
            ]
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            return [0, idle_mask, chosen, requested, 0, 0, 0, 0, 0]

    def _memory_rows(
        self, chosen: int, requested: int, idle_mask: int
    ) -> list[list[int]]:
        row = self._memory_row(chosen, requested, idle_mask)
        if self.group is None:
            return [row]
        gathered = mx.distributed.all_gather(
            mx.array(row, dtype=mx.int64), group=self.group
        )
        return cast(
            list[list[int]], gathered.reshape(self.group.size(), len(row)).tolist()
        )

    def _admit_memory(self, chosen: int, requested: int, idle: list[_Slot]) -> bool:
        if self._memory_limits is None:
            return True
        idle_mask = sum(
            1 << index for index, slot in enumerate(self._slots) if slot in idle
        )

        def agreed(rows: list[list[int]]) -> bool:
            return all(row[0] == 1 and row[1:4] == rows[0][1:4] for row in rows)

        rows = self._memory_rows(chosen, requested, idle_mask)
        if agreed(rows):
            return True
        # Every rank makes this decision from the same gathered rows. Different
        # idle/active ownership cannot be repaired by clearing a local active owner.
        if any(row[1] != idle_mask for row in rows):
            raise RuntimeError("cooperative_memory_owner_state_mismatch")
        for index, slot in enumerate(self._slots):
            if slot in idle and index != chosen and slot.kv_prefix_cache is not None:
                slot.kv_prefix_cache.clear()
        mx.clear_cache()
        rows = self._memory_rows(chosen, requested, idle_mask)
        if agreed(rows):
            return True
        if self._slots[chosen] in idle:
            prefix = self._slots[chosen].kv_prefix_cache
            if prefix is not None:
                prefix.clear()
        mx.clear_cache()
        return agreed(self._memory_rows(chosen, requested, idle_mask))

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
                self._memory_plans[index] = 0
                slot.release_cache_handle()
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
                slot.release_cache_handle()
                self._memory_plans[index] = 0
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
            task = self._queue[0]
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
            chosen = max(range(len(scores)), key=scores.__getitem__)
            try:
                requested = reservation_bytes(
                    len(tokens), task.task_params.max_output_tokens or 0
                )
            except ValueError:
                requested = -1
            if not self._admit_memory(chosen, requested, idle):
                if len(idle) == len(self._slots):
                    self._queue.popleft()
                    self._send_error(
                        task, MemoryError("cooperative_memory_admission_failed")
                    )
                    self._pending.append(
                        (task.task_id, FinishedResponse(task_status=TaskStatus.Failed))
                    )
                    continue
                break
            self._queue.popleft()
            self._memory_plans[chosen] = requested
            slot = self._slots[chosen]
            slot._all_tasks[task.task_id] = task
            slot._queue.append(task)
        result = self._pending
        self._pending = []
        for index, slot in enumerate(self._slots):
            # Runner sends each yielded chunk immediately. Drain earlier slot
            # output before another slot's prefill can interleave and directly
            # send later tokens from that same request.
            for task_id, response in result:
                if isinstance(response, (FinishedResponse, CancelledResponse)):
                    self._retire(task_id)
                yield task_id, response
            result.clear()
            if slot._active is not None or slot._queue:
                result.extend(self._advance(index))
            result.extend(self._pending)
            self._pending = []
        for task_id, response in result:
            if isinstance(response, (FinishedResponse, CancelledResponse)):
                self._retire(task_id)
            yield task_id, response

    def serve_prefill(self, request: PrefillRequest, wfile: BinaryIO) -> None:
        raise ValueError(
            "Cooperative native slots do not support disaggregated prefill"
        )

    def close(self) -> None:
        for slot in self._slots:
            if slot._active is not None:
                slot._active[1].close()
                slot._active = None
            slot.release_cache_handle()
            slot._queue.clear()
            slot._all_tasks.clear()
            if slot.kv_prefix_cache is not None:
                slot.kv_prefix_cache.clear()
            slot.on_cooperative_prefill_progress = None
            slot.parent = None
            slot.close()
        self._slots.clear()
        self._memory_plans = [0, 0]
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
