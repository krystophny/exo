"""Type stubs for mlx_lm.models.deepseek_v41 (the parts exo uses)."""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    o_groups: int
    n_routed_experts: int
    num_experts_per_tok: int
    sliding_window: int
    hc_mult: int
    max_position_embeddings: int
    engram_layer_ids: tuple[int, ...]

class LayerCache:
    window_size: int
    offset: int
    window: Optional[mx.array]
    compress_kv: Optional[mx.array]
    index_k: Optional[mx.array]
    pending_kv: Optional[mx.array]
    pending_gate: Optional[mx.array]

    def __init__(self, window_size: int) -> None: ...
    @property
    def state(self) -> tuple[Optional[mx.array], ...]: ...
    def is_trimmable(self) -> bool: ...
    def empty(self) -> bool: ...
    @property
    def nbytes(self) -> int: ...

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int) -> None: ...

class TextModel(nn.Module):
    args: ModelArgs
    embed_tokens: nn.Embedding
    layers: list[Optional[DecoderLayer]]
    norm: nn.RMSNorm
    pipeline_rank: int
    pipeline_size: int
    start_idx: int
    end_idx: Optional[int]

    def __init__(self, args: ModelArgs) -> None: ...
    @property
    def pipeline_layers(self) -> list[DecoderLayer]: ...
    def pipeline(
        self, group: mx.distributed.Group, split: Optional[list[int]] = None
    ) -> None: ...
    def make_cache(self) -> list[LayerCache]: ...
    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array: ...

class Model(nn.Module):
    args: ModelArgs
    model_type: str
    model: TextModel
    lm_head: nn.Linear

    def __init__(self, config: ModelArgs) -> None: ...
    def __call__(
        self, inputs: mx.array, cache: Optional[list[LayerCache]] = None
    ) -> mx.array: ...
    @property
    def layers(self) -> list[DecoderLayer]: ...
    def make_cache(self) -> list[LayerCache]: ...
    def shard(self, group: Optional[mx.distributed.Group] = None) -> None: ...
    def attach_tokenizer(self, tokenizer: Any) -> None: ...
    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]: ...
