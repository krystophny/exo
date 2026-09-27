"""Native target oracle for two independent MTP/INT8 cache owners."""

import gc
import weakref
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.events import ChunkGenerated, Event
from exo.shared.types.tasks import TaskId
from exo.shared.types.text_generation import (
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    encode_prompt,
)
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.tests.test_mtp import tiny_model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens


@pytest.mark.parametrize("all_accept", [False, True])
def test_native_decode_advances_during_other_prefill(
    monkeypatch: pytest.MonkeyPatch, all_accept: bool
) -> None:
    for key, value in {
        "EXO_COOPERATIVE_SLOTS": "2",
        "EXO_MTP": "1",
        "EXO_MTP_DRAFT_TOKENS": "1",
        "EXO_NO_BATCH": "1",
        "EXO_PREFIX_CACHE_SINGLE_SESSION": "1",
        "EXO_PREFILL_STEP_SIZE": "4",
        "EXO_INTERLEAVED_PREFILL_STEP_SIZE": "4",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_BITS", 8)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_GROUP_SIZE", 64)
    model, tokenizer = tiny_model(all_accept)
    prompts = ["1 2 3 4 5 6", " ".join(str(i % 50) for i in range(100))]
    expected: list[list[int]] = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        oracle: list[int] = []
        for token, _ in generate_step(
            mx.array(tokens),
            model,
            max_tokens=20,
            kv_bits=8,
            quantized_kv_start=0,
            prefill_step_size=4,
        ):
            oracle.append(int(token))
            if int(token) in (tokenizer.eos_token_ids or []):
                break
        expected.append(oracle)
    owners = [KVPrefixCache(None), KVPrefixCache(None)]
    task = TextGenerationTaskParams(
        model=ModelId("test/glm"), input=[], temperature=0, max_output_tokens=20
    )
    first = mlx_generate(model, tokenizer, task, prompts[0], owners[0], None)
    actual_first = [next(first).token]
    overlap: list[int] = []

    def advance_first(processed: int, total: int) -> None:
        if processed < total:
            response = next(first, None)
            if response is not None:
                actual_first.append(response.token)
                overlap.append(processed)

    actual_second = list(
        mlx_generate(
            model,
            tokenizer,
            task,
            prompts[1],
            owners[1],
            None,
            on_prefill_progress=advance_first,
        )
    )
    actual_first.extend(item.token for item in first)
    assert overlap, (
        "no first-session tokens generated while second prompt was prefilling"
    )
    assert actual_first == expected[0]
    assert [item.token for item in actual_second] == expected[1]
    assert owners[0].caches[0] is not owners[1].caches[0]
    assert cache_length(owners[0].caches[0]) == 6 + len(actual_first) - 1
    assert cache_length(owners[1].caches[0]) == 100 + len(actual_second) - 1


def test_engine_two_slots_emit_during_competing_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exo.shared.types.chunks import GenerationChunk, PrefillProgressChunk
    from exo.shared.types.common import CommandId
    from exo.shared.types.tasks import TextGeneration
    from exo.shared.types.worker.instances import InstanceId
    from exo.shared.types.worker.runner_response import FinishedResponse
    from exo.worker.runner.llm_inference.cooperative_generator import (
        CooperativeGenerator,
    )

    for key, value in {
        "EXO_COOPERATIVE_SLOTS": "2",
        "EXO_MTP": "1",
        "EXO_MTP_DRAFT_TOKENS": "1",
        "EXO_NO_BATCH": "1",
        "EXO_PREFIX_CACHE_SINGLE_SESSION": "1",
        "EXO_PREFILL_STEP_SIZE": "4",
        "EXO_INTERLEAVED_PREFILL_STEP_SIZE": "4",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_BITS", 8)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_GROUP_SIZE", 64)

    def render(_tokenizer: TokenizerWrapper, task: TextGenerationTaskParams) -> str:
        return task.instructions or ""

    monkeypatch.setattr(
        "exo.worker.runner.llm_inference.batch_generator.apply_chat_template", render
    )
    model, tokenizer = tiny_model(True)

    class Receiver(MpReceiver[TaskId]):
        def __init__(self) -> None:
            self.pending: list[TaskId] = []

        def collect(self) -> list[TaskId]:
            result, self.pending = self.pending, []
            return result

    class Sender(MpSender[Event]):
        def __init__(self) -> None:
            self.events: list[ChunkGenerated] = []

        def send(self, item: Event) -> None:
            assert isinstance(item, ChunkGenerated)
            self.events.append(item)

    sender = Sender()
    receiver = Receiver()
    engine = CooperativeGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        tool_parser=None,
        kv_prefix_cache=KVPrefixCache(None),
        model_id=ModelId("test/glm"),
        device_rank=0,
        cancel_receiver=receiver,
        event_sender=sender,
    )
    first = TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/glm"),
            input=[],
            instructions=InputMessageContent("1 2 3 4"),
            bench=True,
            use_prefix_cache=True,
            temperature=0,
            max_output_tokens=40,
        ),
    )
    second = TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/glm"),
            input=[],
            instructions=InputMessageContent(" ".join(str(i % 50) for i in range(100))),
            bench=True,
            use_prefix_cache=True,
            temperature=0,
            max_output_tokens=20,
        ),
    )
    engine.submit(first)
    first_output = list(engine.step())
    assert first_output
    engine.submit(second)
    second_step = list(engine.step())
    progresses = [
        i
        for i, e in enumerate(sender.events)
        if e.command_id == second.command_id
        and isinstance(e.chunk, PrefillProgressChunk)
    ]
    overlapping = [
        i
        for i, e in enumerate(sender.events)
        if e.command_id == first.command_id and isinstance(e.chunk, GenerationChunk)
    ]
    assert progresses and overlapping
    assert min(progresses) < min(overlapping) < max(progresses)
    finished = set(
        task_id
        for task_id, response in first_output + second_step
        if isinstance(response, FinishedResponse)
    )
    for _ in range(100):
        finished.update(
            task_id
            for task_id, response in engine.step()
            if isinstance(response, FinishedResponse)
        )
        if len(finished) == 2:
            break
    assert finished == {first.task_id, second.task_id}
    from exo.shared.types.chunks import TokenChunk
    from exo.shared.types.worker.runner_response import CancelledResponse

    warm = first.model_copy(
        update={
            "task_id": TaskId(),
            "command_id": CommandId(),
            "task_params": first.task_params.model_copy(
                update={"max_output_tokens": 4}
            ),
        }
    )
    engine.submit(warm)
    warm_chunks: list[TokenChunk] = []
    for _ in range(20):
        outputs = list(engine.step())
        warm_chunks.extend(
            chunk
            for tid, chunk in outputs
            if tid == warm.task_id and isinstance(chunk, TokenChunk)
        )
        if any(
            tid == warm.task_id and isinstance(chunk, FinishedResponse)
            for tid, chunk in outputs
        ):
            break
    assert any(
        chunk.usage is not None
        and chunk.usage.prompt_tokens_details is not None
        and chunk.usage.prompt_tokens_details.cached_tokens > 0
        for chunk in warm_chunks
    )
    victim = first.model_copy(update={"task_id": TaskId(), "command_id": CommandId()})
    survivor = second.model_copy(
        update={"task_id": TaskId(), "command_id": CommandId()}
    )
    engine.submit(victim)
    list(engine.step())
    engine.submit(survivor)
    list(engine.step())
    receiver.pending.append(victim.task_id)
    cancellation_results: list[
        tuple[TaskId, GenerationChunk | CancelledResponse | FinishedResponse]
    ] = []
    for _ in range(100):
        outputs = list(engine.step())
        cancellation_results.extend(outputs)
        if any(
            tid == survivor.task_id and isinstance(chunk, FinishedResponse)
            for tid, chunk in outputs
        ):
            break
    assert (
        sum(
            tid == victim.task_id and isinstance(chunk, CancelledResponse)
            for tid, chunk in cancellation_results
        )
        == 1
    )
    assert not any(
        tid == victim.task_id and isinstance(chunk, FinishedResponse)
        for tid, chunk in cancellation_results
    )
    assert any(
        tid == survivor.task_id and isinstance(chunk, FinishedResponse)
        for tid, chunk in cancellation_results
    )
    # Cancellation for a queued task is shared across both slots; it must
    # neither close the other active generator nor run the canceled prompt.
    active_a = first.model_copy(
        update={
            "task_id": TaskId(),
            "command_id": CommandId(),
            "task_params": first.task_params.model_copy(
                update={"max_output_tokens": 16}
            ),
        }
    )
    active_b = active_a.model_copy(
        update={"task_id": TaskId(), "command_id": CommandId()}
    )
    queued = active_a.model_copy(
        update={"task_id": TaskId(), "command_id": CommandId()}
    )
    engine.submit(active_a)
    engine.submit(active_b)
    queued_results = list(engine.step())
    engine.submit(queued)
    receiver.pending.extend([queued.task_id, active_a.task_id])
    for _ in range(50):
        queued_results.extend(engine.step())
        if any(
            tid == active_b.task_id and isinstance(chunk, FinishedResponse)
            for tid, chunk in queued_results
        ):
            break
    assert (
        sum(
            tid == active_b.task_id and isinstance(chunk, TokenChunk)
            for tid, chunk in queued_results
        )
        + sum(
            event.command_id == active_b.command_id
            and isinstance(event.chunk, TokenChunk)
            for event in sender.events
        )
        == 16
    )
    assert any(
        tid == active_b.task_id and isinstance(chunk, FinishedResponse)
        for tid, chunk in queued_results
    )
    for canceled in [queued, active_a]:
        assert (
            sum(
                tid == canceled.task_id and isinstance(chunk, CancelledResponse)
                for tid, chunk in queued_results
            )
            == 1
        )
        assert not any(
            tid == canceled.task_id and isinstance(chunk, FinishedResponse)
            for tid, chunk in queued_results
        )
    assert not any(
        tid == queued.task_id and isinstance(chunk, TokenChunk)
        for tid, chunk in queued_results
    )

    # A completed request must release its full task/transcript object even
    # while both native prefix owners remain warm for later conversations.
    retired_references: list[weakref.ReferenceType[TextGeneration]] = []
    for _ in range(20):
        transient = first.model_copy(
            update={
                "task_id": TaskId(),
                "command_id": CommandId(),
                "task_params": first.task_params.model_copy(
                    update={"max_output_tokens": 1}
                ),
            }
        )
        retired_references.append(weakref.ref(transient))
        engine.submit(transient)
        for _ in range(10):
            transient_outputs = list(engine.step())
            if any(
                tid == transient.task_id and isinstance(chunk, FinishedResponse)
                for tid, chunk in transient_outputs
            ):
                break
        else:
            pytest.fail("repeated request never completed")
        del transient
    gc.collect()
    assert all(reference() is None for reference in retired_references), (
        "completed prompt objects retained"
    )

    retained_owners = [slot.kv_prefix_cache for slot in engine._slots]  # pyright: ignore[reportPrivateUsage]
    retained_slots = list(engine._slots)  # pyright: ignore[reportPrivateUsage]
    model_reference = weakref.ref(model)
    del model
    assert model_reference() is not None
    engine.close()
    gc.collect()
    assert model_reference() is None, "engine close retained model residency"
    assert not engine._slots  # pyright: ignore[reportPrivateUsage]
    assert all(owner is not None and not owner.caches for owner in retained_owners)
    assert all(
        slot.parent is None and slot.on_cooperative_prefill_progress is None
        for slot in retained_slots
    )
    assert all(not hasattr(slot, "model") for slot in retained_slots)


def test_request_local_sampling_matches_native_and_interleaving() -> None:
    from mlx_lm.sample_utils import make_sampler

    from exo.worker.engines.mlx.session_sampling import make_session_sampler

    logits = mx.log(mx.array([[0.1, 0.2, 0.3, 0.4]]))
    mx.random.seed(123)
    native = make_sampler(temp=1, top_p=0.95, min_p=0.05, top_k=3)
    expected = [int(native(logits).item()) for _ in range(50)]
    isolated = make_session_sampler(123, 1, 0.95, 0.05, 3)
    other = make_session_sampler(456, 1, 0.9, 0.05, 2)
    actual: list[int] = []
    for _ in range(50):
        other(logits).item()
        mx.random.seed(987)
        actual.append(int(isolated(logits).item()))
    assert actual == expected


def test_interleaved_streams_preserve_caller_wired_limit() -> None:
    from mlx_lm.generate import stream_generate

    model, tokenizer = tiny_model(True)
    limit = 8 * 1024**3
    previous = mx.set_wired_limit(limit)
    first = stream_generate(
        model, tokenizer, [1, 2, 3], max_tokens=8, manage_wired_limit=False
    )
    second = stream_generate(
        model, tokenizer, [4, 5, 6], max_tokens=8, manage_wired_limit=False
    )
    try:
        next(first)
        next(second)
        first.close()  # Deliberately non-LIFO: second remains active.
        assert mx.set_wired_limit(limit) == limit
        next(second)
        second.close()
        assert mx.set_wired_limit(limit) == limit
    finally:
        first.close()
        second.close()
        mx.set_wired_limit(previous)


def test_owner_turn_survives_interleaved_computor_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three cooperative slots, one active generation at a time (as a real
    Slopgate deployment would run it): owner turn 1, then an interleaved
    computor request, then owner turn 2. Turn 2 must reuse the owner's own
    cached prefix (observable via the emitted cached_tokens usage stat and
    directly via prefix_match_length), and both requests' outputs must match
    standalone oracle runs of the same model.
    """
    from exo.shared.types.chunks import TokenChunk
    from exo.shared.types.common import CommandId
    from exo.shared.types.tasks import TextGeneration
    from exo.shared.types.worker.instances import InstanceId
    from exo.shared.types.worker.runner_response import FinishedResponse
    from exo.worker.runner.llm_inference.cooperative_generator import (
        CooperativeGenerator,
        _OwnerPrefixCache,  # pyright: ignore[reportPrivateUsage]
    )

    for key, value in {
        "EXO_COOPERATIVE_SLOTS": "3",
        "EXO_COOPERATIVE_ACTIVE": "1",
        "EXO_MTP": "1",
        "EXO_MTP_DRAFT_TOKENS": "1",
        "EXO_NO_BATCH": "1",
        "EXO_PREFIX_CACHE_SINGLE_SESSION": "1",
        "EXO_PREFILL_STEP_SIZE": "4",
        "EXO_INTERLEAVED_PREFILL_STEP_SIZE": "4",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_BITS", 8)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_GROUP_SIZE", 64)

    def render(_tokenizer: TokenizerWrapper, task: TextGenerationTaskParams) -> str:
        return task.instructions or ""

    monkeypatch.setattr(
        "exo.worker.runner.llm_inference.batch_generator.apply_chat_template", render
    )
    model, tokenizer = tiny_model(True)

    def oracle(prompt: str) -> list[int]:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        result: list[int] = []
        for token, _ in generate_step(
            mx.array(tokens),
            model,
            max_tokens=8,
            kv_bits=8,
            quantized_kv_start=0,
            prefill_step_size=4,
        ):
            result.append(int(token))
            if int(token) in (tokenizer.eos_token_ids or []):
                break
        return result

    class Receiver(MpReceiver[TaskId]):
        def __init__(self) -> None:
            self.pending: list[TaskId] = []

        def collect(self) -> list[TaskId]:
            result, self.pending = self.pending, []
            return result

    class Sender(MpSender[Event]):
        def __init__(self) -> None:
            self.events: list[ChunkGenerated] = []

        def send(self, item: Event) -> None:
            assert isinstance(item, ChunkGenerated)
            self.events.append(item)

    engine = CooperativeGenerator(
        model=model,
        tokenizer=tokenizer,
        group=None,
        tool_parser=None,
        kv_prefix_cache=KVPrefixCache(None),
        model_id=ModelId("test/glm"),
        device_rank=0,
        cancel_receiver=Receiver(),
        event_sender=Sender(),
        num_slots=3,
    )

    def _count_busy_slots(coop: CooperativeGenerator) -> int:
        slots = coop._slots  # pyright: ignore[reportPrivateUsage]
        return sum(
            1
            for slot in slots
            if slot._active is not None  # pyright: ignore[reportPrivateUsage]
            or slot._queue  # pyright: ignore[reportPrivateUsage]
        )

    def run_to_completion(task: TextGeneration) -> list[TokenChunk]:
        chunks: list[TokenChunk] = []
        for _ in range(200):
            outputs = list(engine.step())
            busy = _count_busy_slots(engine)
            assert busy <= 1, "EXO_COOPERATIVE_ACTIVE=1 must never admit a 2nd slot"
            chunks.extend(
                chunk
                for tid, chunk in outputs
                if tid == task.task_id and isinstance(chunk, TokenChunk)
            )
            if any(
                tid == task.task_id and isinstance(chunk, FinishedResponse)
                for tid, chunk in outputs
            ):
                break
        return chunks

    owner_key = "owner-session-1"
    owner_prompt_1 = "1 2 3 4 5 6"
    owner_turn1 = TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/glm"),
            input=[],
            instructions=InputMessageContent(owner_prompt_1),
            bench=True,
            use_prefix_cache=True,
            temperature=0,
            max_output_tokens=8,
            session_class="owner",
            session_key=owner_key,
        ),
    )
    engine.submit(owner_turn1)
    owner1_chunks = run_to_completion(owner_turn1)
    assert [c.token_id for c in owner1_chunks] == oracle(owner_prompt_1)
    assert (
        engine._slot_last_key[0]  # pyright: ignore[reportPrivateUsage]
        == owner_key
    )  # owner turn 1 landed on slot 0

    owner_prefix = cast(
        _OwnerPrefixCache,
        engine._slots[0].kv_prefix_cache,  # pyright: ignore[reportPrivateUsage]
    )
    owner1_tokens = fix_unmatched_think_end_tokens(
        encode_prompt(tokenizer, owner_prompt_1), tokenizer
    )

    computor_prompt = " ".join(str(i % 50) for i in range(60))
    computor_task = TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/glm"),
            input=[],
            instructions=InputMessageContent(computor_prompt),
            bench=True,
            use_prefix_cache=True,
            temperature=0,
            max_output_tokens=8,
            session_class="computor",
        ),
    )
    engine.submit(computor_task)
    computor_chunks = run_to_completion(computor_task)
    assert [c.token_id for c in computor_chunks] == oracle(computor_prompt)
    # The computor request must never have touched the owner's slot: its
    # cached prefix for turn 1's prompt is still fully intact.
    assert owner_prefix.prefix_match_length(owner1_tokens) == len(owner1_tokens)
    # computor never records a key
    assert engine._slot_last_key[2] is None  # pyright: ignore[reportPrivateUsage]

    owner_prompt_2 = f"{owner_prompt_1} 7 8 9"
    owner2_tokens = fix_unmatched_think_end_tokens(
        encode_prompt(tokenizer, owner_prompt_2), tokenizer
    )
    # Cache-hit oracle: before turn 2 is even admitted, the owner's own slot
    # already matches the whole of turn 1's prompt as a strict prefix.
    assert owner_prefix.prefix_match_length(owner2_tokens) == len(owner1_tokens)

    owner_turn2 = TextGeneration(
        instance_id=InstanceId(),
        command_id=CommandId(),
        task_params=TextGenerationTaskParams(
            model=ModelId("test/glm"),
            input=[],
            instructions=InputMessageContent(owner_prompt_2),
            bench=True,
            use_prefix_cache=True,
            temperature=0,
            max_output_tokens=8,
            session_class="owner",
            session_key=owner_key,
        ),
    )
    engine.submit(owner_turn2)
    owner2_chunks = run_to_completion(owner_turn2)
    assert [c.token_id for c in owner2_chunks] == oracle(owner_prompt_2)
    # turn 2 stayed on its own slot
    turn2_key = engine._slot_last_key[0]  # pyright: ignore[reportPrivateUsage]
    assert turn2_key == owner_key
    # The prefill actually reused the cached prefix (observable cache hit):
    # the engine trims the last matched token to keep decode state exact,
    # so the reported cache hit is turn 1's length minus a small margin.
    assert any(
        chunk.usage is not None
        and chunk.usage.prompt_tokens_details is not None
        and chunk.usage.prompt_tokens_details.cached_tokens
        >= len(owner1_tokens) - 2
        for chunk in owner2_chunks
    )
