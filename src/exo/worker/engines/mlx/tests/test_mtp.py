import dataclasses
import json
from typing import cast

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.generate import generate_step
from mlx_lm.models import glm_moe_dsa
from mlx_lm.models.deepseek_v32 import DeepseekV32MoE, ModelArgs, MtpModule
from mlx_lm.tokenizer_utils import TokenizerWrapper
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.cache import KVPrefixCache, cache_length
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.types import Model


def tiny_model(all_accept: bool):
    config = dataclasses.asdict(ModelArgs())
    config.update(
        model_type="glm_moe_dsa",
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        n_shared_experts=1,
        kv_lora_rank=64,
        q_lora_rank=64,
        qk_rope_head_dim=64,
        v_head_dim=64,
        qk_nope_head_dim=64,
        index_head_dim=64,
        index_n_heads=16,
        index_topk=8,
        indexer_types=["full", "full", "shared", "shared"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        num_nextn_predict_layers=1,
    )
    mx.random.seed(4)
    model = glm_moe_dsa.Model(glm_moe_dsa.ModelArgs.from_dict(config))
    model.mtp = MtpModule(model.args)
    for layer in [*model.model.layers, model.mtp.layer]:
        assert isinstance(layer.mlp, DeepseekV32MoE)
        layer.mlp.gate.weight = mx.random.normal(layer.mlp.gate.weight.shape) * 0.03
    if all_accept:
        model.lm_head.weight = mx.zeros_like(model.lm_head.weight)
    backend = Tokenizer.from_str(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {str(i): i for i in range(64)},
                    "unk_token": "63",
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        )
    )
    tokenizer = TokenizerWrapper(
        PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="63")
    )
    return cast(Model, cast(nn.Module, model)), tokenizer


@pytest.fixture(autouse=True)
def mtp_environment(monkeypatch: pytest.MonkeyPatch):
    for key, value in {
        "EXO_MTP": "1",
        "EXO_MTP_DRAFT_TOKENS": "1",
        "EXO_NO_BATCH": "1",
        "EXO_PREFIX_CACHE_SINGLE_SESSION": "1",
        "EXO_PREFILL_STEP_SIZE": "4",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_BITS", 8)
    monkeypatch.setattr("exo.worker.engines.mlx.generator.generate.KV_GROUP_SIZE", 64)


@pytest.mark.parametrize("all_accept", [False, True])
def test_mtp_prefix_reuse_matches_cold_target(all_accept: bool):
    model, tokenizer = tiny_model(all_accept)
    owner = KVPrefixCache(None)
    task = TextGenerationTaskParams(
        model=ModelId("test/glm"), input=[], temperature=0, max_output_tokens=7
    )
    progress: list[tuple[int, int]] = []
    for prompt in ("1 2 3 4 5 6 7 8 9", "1 2 3 4 5 6 7 8 9", "1 2 3 4 5 6 11 12 13"):
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        expected: list[int] = []
        for token, _ in generate_step(
            mx.array(tokens),
            model,
            max_tokens=7,
            kv_bits=8,
            quantized_kv_start=0,
            prefill_step_size=4,
        ):
            expected.append(int(token))
            if int(token) in (tokenizer.eos_token_ids or []):
                break
        actual = list(
            mlx_generate(
                model,
                tokenizer,
                task,
                prompt,
                owner,
                None,
                on_prefill_progress=lambda n, total: progress.append((n, total)),
            )
        )
        assert [item.token for item in actual] == expected
        assert actual[-1].usage is not None
        assert actual[-1].usage.prompt_tokens == len(tokens)
        assert cache_length(owner.caches[0]) == len(tokens) + len(expected) - 1
    assert progress[-1][0] == progress[-1][1]


def test_mtp_stop_and_cancel_keep_followup_cache_correct():
    model, tokenizer = tiny_model(True)
    owner = KVPrefixCache(None)
    task = TextGenerationTaskParams(
        model=ModelId("test/glm"),
        input=[],
        temperature=0,
        max_output_tokens=12,
        stop="0",
    )
    output = list(mlx_generate(model, tokenizer, task, "1 2 3", owner, None))
    assert len(output) == 1
    assert output[0].finish_reason == "stop"
    assert cache_length(owner.caches[0]) == 3
    task = task.model_copy(update={"stop": None})
    responses = mlx_generate(model, tokenizer, task, "1 2 3 0 4", owner, None)
    next(responses)
    responses.close()
    assert cache_length(owner.caches[0]) == 5
    warm = list(mlx_generate(model, tokenizer, task, "1 2 3 0 4 0 5", owner, None))
    cold = list(mlx_generate(model, tokenizer, task, "1 2 3 0 4 0 5", None, None))
    assert [out.token for out in warm] == [out.token for out in cold]
