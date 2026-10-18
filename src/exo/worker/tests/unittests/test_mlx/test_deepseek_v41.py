"""DeepSeek V4.1 in exo without the real weights: the placement-to-pipeline
contract, sharding on a single rank and cache snapshots on a tiny random
model built from mlx-lm's own module."""

import tomllib
from collections.abc import Generator
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest
from mlx_lm.models import deepseek_v41
from mlx_lm.models.pipeline import PipelineMixin

from exo.master.placement_utils import get_shard_assignments_for_pipeline_parallel
from exo.master.tests.conftest import create_node_memory
from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ModelCard
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.topology import Cycle
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import PipelineShardMetadata
from exo.worker.engines.mlx.auto_parallel import (
    native_pipeline_split,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.cache import (
    has_non_kv_caches,
    make_kv_cache,
    snapshot_ssm_states,
    trim_cache,
)
from exo.worker.engines.mlx.types import Model

_CARD_PATH = (
    Path(RESOURCES_DIR)
    / "inference_model_cards"
    / "deepseek-ai--DeepSeek-V4.1-Flash-1017.toml"
)


def _card() -> ModelCard:
    return ModelCard.model_validate(tomllib.loads(_CARD_PATH.read_text()))


class _Group:
    def __init__(self, rank: int, size: int) -> None:
        self._rank, self._size = rank, size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


class _Stages(PipelineMixin):
    """mlx-lm's PipelineMixin on bare layer numbers: the layer ranges the
    real model would keep."""

    def __init__(self, n_layers: int) -> None:
        super().__init__()
        self.layers = list(range(n_layers))


def _mlx_lm_range(split: list[int], rank: int, n_layers: int) -> tuple[int, int]:
    stages = _Stages(n_layers)
    group = cast(mx.distributed.Group, cast(object, _Group(rank, len(split))))
    stages.pipeline(group, split)
    assert stages.end_idx is not None
    return stages.start_idx, stages.end_idx


@pytest.mark.parametrize("world_size", [2, 3])
def test_placement_and_mlx_lm_pipeline_agree_on_every_rank(world_size: int) -> None:
    card = _card()
    nodes = [NodeId() for _ in range(world_size)]
    # Two nodes need over 300 GB each: whichever holds layer 14's n-gram table
    # must also hold the 25 layers after it. 320 GB lets both sizes place.
    memory = {node: create_node_memory(320_000_000_000) for node in nodes}
    assignments = get_shard_assignments_for_pipeline_parallel(
        card, Cycle(node_ids=nodes), memory
    )
    shards = [
        cast(PipelineShardMetadata, assignments.runner_to_shard[runner])
        for runner in (assignments.node_to_runner[node] for node in nodes)
    ]
    for shard in shards:
        split = native_pipeline_split(
            shard.device_rank,
            world_size,
            shard.start_layer,
            shard.end_layer,
            shard.n_layers,
        )
        assert _mlx_lm_range(split, shard.device_rank, card.n_layers) == (
            shard.start_layer,
            shard.end_layer,
        )
    by_rank = sorted(shards, key=lambda s: s.device_rank)
    assert by_rank[-1].start_layer == 0
    assert by_rank[0].end_layer == card.n_layers


def test_native_split_rejects_a_forward_ordered_pipeline() -> None:
    # Rank 0 holding the first layers is exo's usual order, not mlx-lm's.
    with pytest.raises(ValueError, match="reversed pipeline"):
        native_pipeline_split(0, 3, 0, 14, 40)
    with pytest.raises(ValueError, match="reversed pipeline"):
        native_pipeline_split(2, 3, 26, 40, 40)


@pytest.mark.parametrize(
    ("rank", "size", "start", "end"),
    [(0, 1, 0, 40), (0, 2, 20, 40), (1, 2, 0, 20), (1, 3, 9, 25), (2, 3, 0, 9)],
)
def test_native_split_reproduces_the_assigned_range(
    rank: int, size: int, start: int, end: int
) -> None:
    split = native_pipeline_split(rank, size, start, end, 40)
    assert len(split) == size and all(part > 0 for part in split)
    assert _mlx_lm_range(split, rank, 40) == (start, end)


def _tiny_model() -> deepseek_v41.Model:
    # The configuration of mlx-lm's own test_models test_deepseek_v41.
    args = deepseek_v41.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v41",
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 6,
            "num_attention_heads": 4,
            "head_dim": 32,
            "q_lora_rank": 32,
            "qk_rope_head_dim": 8,
            "o_groups": 2,
            "o_lora_rank": 16,
            "moe_intermediate_size": 32,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "sliding_window": 4,
            "compress_ratios": (0, 2, 2, 2, 1, 1),
            "kv_source_layer_ids": (1, 4),
            "index_source_layer_ids": (1, 2, 4, 5),
            "index_n_heads": 4,
            "index_head_dim": 32,
            "index_topk": 3,
            "candidate_source_layer_id": 4,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 2,
        }
    )
    mx.random.seed(0)
    model = deepseek_v41.Model(args)
    mx.eval(model.parameters())
    return model


def _drain(
    loader: Generator[ModelLoadingResponse, None, object],
) -> tuple[object, int]:
    steps = 0
    while True:
        try:
            next(loader)
            steps += 1
        except StopIteration as done:
            return cast(object, done.value), steps


_PROMPT = mx.array([[3, 17, 42, 5, 99, 7, 11, 64, 2, 8]])


def test_single_rank_pipeline_and_tensor_loading_keep_the_outputs() -> None:
    model = _tiny_model()
    expected = model(_PROMPT)
    group = mx.distributed.init()
    assert group.size() == 1

    shard = PipelineShardMetadata(
        model_card=_card(),
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=6,
        n_layers=6,
    )
    piped, steps = _drain(pipeline_auto_parallel(model, group, shard))
    assert piped is model and steps == 6
    assert mx.allclose(model(_PROMPT), expected, atol=1e-5).item()

    other = _tiny_model()
    sharded, steps = _drain(tensor_auto_parallel(other, group))
    assert steps == 6
    assert mx.allclose(
        cast(deepseek_v41.Model, sharded)(_PROMPT), expected, atol=1e-5
    ).item()


def test_cache_snapshot_rolls_back_a_v41_cache() -> None:
    model = _tiny_model()
    cache = make_kv_cache(cast(Model, cast(object, model)))
    assert all(isinstance(c, deepseek_v41.LayerCache) for c in cache)
    assert has_non_kv_caches(cache)

    model(_PROMPT, cache=list(cache))  # pyright: ignore[reportArgumentType]
    snapshot = snapshot_ssm_states(cache)
    assert snapshot.token_count == _PROMPT.shape[1]
    step = mx.array([[21]])
    first = model(step, cache=list(cache))  # pyright: ignore[reportArgumentType]
    # Run on past the window so evicted rows and closed groups would differ.
    for token in (22, 23, 24, 25, 26):
        model(mx.array([[token]]), cache=list(cache))  # pyright: ignore[reportArgumentType]

    restored = list(cache)
    trim_cache(restored, 6, snapshot)
    again = model(step, cache=restored)  # pyright: ignore[reportArgumentType]
    assert mx.array_equal(first, again).item()


def test_trimming_a_v41_cache_without_a_snapshot_fails_loudly() -> None:
    model = _tiny_model()
    cache = list(make_kv_cache(cast(Model, cast(object, model))))
    model(_PROMPT, cache=cache)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="snapshot"):
        trim_cache(cache, 1)


def test_card_storage_is_consistent() -> None:
    card = _card()
    per_layer = card.layer_storage_bytes()
    assert sum(per_layer) == card.storage_size.in_bytes
    heavy = sorted(range(card.n_layers), key=lambda i: per_layer[i])[-2:]
    assert sorted(heavy) == [1, 14]
    assert Memory.from_bytes(per_layer[1]).in_gb > 100
