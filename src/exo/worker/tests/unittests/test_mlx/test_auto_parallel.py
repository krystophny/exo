import json
import multiprocessing as mp
import os
import tempfile
from typing import Any

import mlx.core as mx
import mlx.nn as mlx_nn
import pytest

from exo.worker.engines.mlx.auto_parallel import (
    CustomMlxLayer,
    PipelineFirstLayer,
    PipelineLastLayer,
    patch_pipeline_model,
    tensor_auto_parallel,
)
from exo.worker.tests.unittests.test_mlx.conftest import MockLayer


def _finish_tensor_auto_parallel(
    model: mlx_nn.Module,
    group: mx.distributed.Group,
) -> tuple[Any, list[Any]]:
    yielded: list[Any] = []
    generator = tensor_auto_parallel(model, group)
    while True:
        try:
            yielded.append(next(generator))
        except StopIteration as exc:
            return exc.value, yielded


def run_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    result_queue: Any,  # pyright: ignore[reportAny]
) -> None:
    import os

    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    class MockLayerInner(mlx_nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.custom_attr = "test_value"

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            return x * 2

    class MockModel(mlx_nn.Module):
        def __init__(self, layers: list[mlx_nn.Module]) -> None:
            super().__init__()
            self.layers = layers

        def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
            return x

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        mock = MockLayerInner()
        first = PipelineFirstLayer(mock, r=rank, group=group)
        composed = PipelineLastLayer(first, r=rank, s=world_size, group=group)

        # Wrap in a mock model, then wrap in PipelineParallelModel for all_gather
        inner_model = MockModel([composed])
        model = patch_pipeline_model(inner_model, group)

        x = mx.ones((1, 4))
        result = model(x)
        mx.eval(result)
        success = result.shape == x.shape
        result_queue.put((rank, success, result))  # pyright: ignore[reportAny]
    except Exception as e:
        result_queue.put((rank, False, str(e)))  # pyright: ignore[reportAny]


def test_single_wrapper_delegates_attributes() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    assert wrapped.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert wrapped.use_sliding is True  # type: ignore[attr-defined]


def test_composed_wrappers_delegate_attributes() -> None:
    mock = MockLayer()
    group = mx.distributed.init()

    first = PipelineFirstLayer(mock, r=0, group=group)
    composed = PipelineLastLayer(first, r=0, s=1, group=group)

    assert composed.custom_attr == "test_value"  # type: ignore[attr-defined]
    assert composed.use_sliding is True  # type: ignore[attr-defined]


def test_missing_attribute_raises() -> None:
    mock = MockLayer()
    wrapped = CustomMlxLayer(mock)

    with pytest.raises(AttributeError):
        _ = wrapped.nonexistent_attr  # type: ignore[attr-defined]


def test_tensor_auto_parallel_supports_hy3() -> None:
    from mlx_lm.models.hy_v3 import MLP as HYV3MLP
    from mlx_lm.models.hy_v3 import Model as HYV3Model
    from mlx_lm.models.hy_v3 import ModelArgs as HYV3ModelArgs
    from mlx_lm.models.hy_v3 import MoE as HYV3MoE

    group = mx.distributed.init()
    model = HYV3Model(
        HYV3ModelArgs(
            model_type="hy_v3",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=8,
            num_experts=4,
            num_experts_per_tok=2,
            num_shared_experts=1,
            expert_hidden_dim=32,
            first_k_dense_replace=1,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_theta": 10000.0},
        )
    )

    parallel_model, loaded = _finish_tensor_auto_parallel(model, group)

    assert parallel_model is model
    assert len(loaded) == 2
    assert model.model.layers[0].self_attn.n_heads == 8 // group.size()
    assert model.model.layers[0].self_attn.n_kv_heads == max(1, 2 // group.size())
    assert isinstance(model.model.layers[0].mlp, HYV3MLP)
    assert isinstance(model.model.layers[1].mlp, HYV3MoE)
    assert model.model.layers[1].mlp.sharding_group is group


def test_composed_call_works() -> None:
    ctx = mp.get_context("spawn")

    world_size = 2
    base_port = 29500

    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    try:
        result_queue: Any = ctx.Queue()

        processes: list[Any] = []
        for rank in range(world_size):
            p = ctx.Process(
                target=run_pipeline_device,
                args=(rank, world_size, hostfile_path, result_queue),
            )
            p.start()
            processes.append(p)

        for p in processes:  # pyright: ignore[reportAny]
            p.join(timeout=10)  # pyright: ignore[reportAny]

        results: dict[int, Any] = {}
        errors: dict[int, str] = {}
        while not result_queue.empty():  # pyright: ignore[reportAny]
            rank, success, value = result_queue.get()  # pyright: ignore[reportAny]
            if success:
                results[rank] = value
            else:
                errors[rank] = value

        assert len(results) == world_size, (
            f"Expected {world_size} results, got {len(results)}. Errors: {errors}"
        )

        for rank in range(world_size):
            assert rank in results, (
                f"Device {rank} failed: {errors.get(rank, 'unknown')}"
            )
            result_array = results[rank]
            # Both devices see the final result (4.0) after all_gather
            assert (result_array == 4.0).all(), (
                f"Device {rank}: expected 4.0, got {result_array}"
            )
    finally:
        os.unlink(hostfile_path)
