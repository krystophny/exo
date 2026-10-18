import tomllib
from pathlib import Path
from typing import Any

import pytest

from exo.master.placement_utils import (
    allocate_layers_by_storage,
    get_shard_assignments_for_pipeline_parallel,
)
from exo.master.tests.conftest import create_node_memory
from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ConfigData, ModelCard
from exo.shared.types.common import NodeId
from exo.shared.types.topology import Cycle
from exo.shared.types.worker.shards import PipelineShardMetadata

_CARD = (
    Path(RESOURCES_DIR)
    / "inference_model_cards"
    / "deepseek-ai--DeepSeek-V4.1-Flash-1017.toml"
)
_GB = 1_000_000_000

# The fields exo reads from deepseek-ai/DeepSeek-V4.1-Flash-1017's config.json.
_OFFICIAL_CONFIG: dict[str, Any] = {
    "architectures": ["DeepseekV41ForCausalLM"],
    "model_type": "deepseek_v41",
    "image_token_id": 129264,
    "text_config": {
        "model_type": "deepseek_v41_text",
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "num_key_value_heads": 1,
        "max_position_embeddings": 1048576,
        "o_groups": 8,
        "num_attention_heads": 64,
    },
    "vision_config": {"model_type": "deepseek_v41_vision"},
}


def _card() -> ModelCard:
    return ModelCard.model_validate(tomllib.loads(_CARD.read_text()))


def test_card_matches_the_official_config() -> None:
    card = _card()
    config = ConfigData.model_validate(
        _OFFICIAL_CONFIG, context={"model_id": str(card.model_id)}
    )
    assert config.layer_count == card.n_layers == 40
    assert config.hidden_size == card.hidden_size
    assert config.num_key_value_heads == card.num_key_value_heads
    assert config.max_position_embeddings == card.context_length
    assert config.supports_tensor and card.supports_tensor
    # exo runs mlx-lm's own V4.1 pipeline, whichever card the model comes from.
    assert config.pipeline_layers_reversed and card.pipeline_layers_reversed
    # mlx-lm drops the vision tower and exo has no V4.1 encoder for it.
    assert config.vision is None


def test_heavy_layers_must_be_real_layers() -> None:
    raw = tomllib.loads(_CARD.read_text())
    raw["extra_layer_storage"][0]["layer"] = 40
    with pytest.raises(ValueError, match="outside"):
        ModelCard.model_validate(raw)


def test_storage_allocation_balances_heavy_layers() -> None:
    layer_bytes = [10] * 8
    layer_bytes[1] = 100
    # The heavy layer fills the first node; the light ones go to the second.
    assert allocate_layers_by_storage(layer_bytes, [120, 120]) == [2, 6]
    # Uniform layers on equal nodes split evenly.
    assert allocate_layers_by_storage([5] * 9, [1, 1, 1]) == [3, 3, 3]


def _place(node_gb: list[int]) -> list[PipelineShardMetadata]:
    card = _card()
    nodes = [NodeId() for _ in node_gb]
    memory = {
        node: create_node_memory(gb * _GB)
        for node, gb in zip(nodes, node_gb, strict=True)
    }
    assignments = get_shard_assignments_for_pipeline_parallel(
        card, Cycle(node_ids=nodes), memory
    )
    shards = [
        assignments.runner_to_shard[assignments.node_to_runner[node]] for node in nodes
    ]
    assert all(isinstance(shard, PipelineShardMetadata) for shard in shards)
    return [shard for shard in shards if isinstance(shard, PipelineShardMetadata)]


def test_three_256gb_macs_place_the_native_checkpoint_in_reverse() -> None:
    # About 250 GB of each Mac is available to exo once nothing else runs.
    shards = _place([250, 250, 250])
    card = shards[0].model_card
    by_rank = sorted(shards, key=lambda s: s.device_rank)
    # Reversed: the last rank starts at layer 0, rank 0 ends at the last layer.
    assert by_rank[2].start_layer == 0
    assert by_rank[1].start_layer == by_rank[2].end_layer
    assert by_rank[0].start_layer == by_rank[1].end_layer
    assert by_rank[0].end_layer == card.n_layers
    loads = [card.shard_storage_bytes(s.start_layer, s.end_layer) for s in by_rank]
    assert all(load <= 250 * _GB for load in loads)
    # The two 101 GB n-gram tables land on different Macs.
    owner = {
        layer: next(
            s.device_rank for s in shards if s.start_layer <= layer < s.end_layer
        )
        for layer in (1, 14)
    }
    assert owner[1] != owner[14]
    # Balanced well below the even 13/13/14 layer split's worst Mac.
    assert max(loads) < 200 * _GB


def test_two_256gb_macs_cannot_hold_the_native_checkpoint() -> None:
    with pytest.raises(ValueError, match="insufficient memory"):
        _place([250, 250])
