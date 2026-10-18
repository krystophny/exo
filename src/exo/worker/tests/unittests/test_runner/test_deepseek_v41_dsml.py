"""DeepSeek V4.1 prompt encoding and DSML tool-call parsing.

The golden prompts in fixtures/deepseek_v41 are DeepSeek's own encoding tests
(deepseek-ai/DeepSeek-V4.1-Flash-1017, encoding/tests, MIT License).
"""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import (
    InputMessage,
    InputMessageContent,
    TextGenerationTaskParams,
)
from exo.shared.types.worker.runner_response import (
    GenerationResponse,
    ToolCallResponse,
)
from exo.worker.engines.mlx.utils_mlx import render_chat_template
from exo.worker.engines.mlx.vendor import deepseek_v41_encoding as v41
from exo.worker.engines.mlx.vendor.dsml_encoding import parse_dsml_output
from exo.worker.runner.llm_inference.model_output_parsers import (
    parse_deepseek_v4,
    parse_deepseek_v41,
)

_FIXTURES = Path(__file__).parent / "fixtures" / "deepseek_v41"
_MODEL = ModelId("deepseek-ai/DeepSeek-V4.1-Flash-1017")
_DSML = "｜DSML｜"

_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


def _simulate(texts: list[str]) -> Generator[GenerationResponse]:
    for i, text in enumerate(texts):
        yield GenerationResponse(
            text=text,
            token=i,
            finish_reason="stop" if i == len(texts) - 1 else None,
            usage=None,
        )


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _official_tool_call_block(calls: list[dict[str, Any]]) -> str:
    """The DSML block DeepSeek's own encoder writes for an assistant turn."""
    rendered = cast(
        str,
        v41.encode_messages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": calls},
            ],
            thinking_mode="chat",
        ),
    )
    start = rendered.index(f"<{_DSML} calls>")
    end = rendered.index(f"</{_DSML} calls>") + len(f"</{_DSML} calls>")
    return rendered[start:end]


@pytest.mark.parametrize("case", ["1", "3"])
def test_vendored_encoder_reproduces_deepseek_golden_prompts(case: str) -> None:
    [loaded] = v41.load_cases(str(_FIXTURES / f"test_input_{case}.json"))
    # DeepSeek's own golden test encodes with chat as the default mode.
    prompt = v41.encode_case(loaded, "chat")[0]
    assert prompt == (_FIXTURES / f"test_output_{case}.txt").read_text()


def test_exo_prompt_matches_official_encoder_with_tools() -> None:
    task = TextGenerationTaskParams(
        model=_MODEL,
        input=[
            InputMessage(role="user", content=InputMessageContent("Weather in Graz?"))
        ],
        tools=[_WEATHER_TOOL],
        reasoning_effort="max",
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Weather in Graz?"},
    ]
    prompt = render_chat_template(
        cast(TokenizerWrapper, object()), [dict(m) for m in messages], task
    )
    expected = cast(
        str,
        v41.encode_messages(
            [{**messages[0], "tools": [_WEATHER_TOOL]}, messages[1]],
            thinking_mode="thinking",
            reasoning_effort="max",
        ),
    )
    assert prompt == expected
    assert "Reasoning Effort: 100" in prompt
    assert f"<{_DSML} calls>" in prompt


def test_exo_prompt_chat_mode_maps_developer_to_system() -> None:
    task = TextGenerationTaskParams(
        model=_MODEL,
        input=[InputMessage(role="user", content=InputMessageContent("hi"))],
        enable_thinking=False,
    )
    prompt = render_chat_template(
        cast(TokenizerWrapper, object()),
        [{"role": "developer", "content": "Rules."}, {"role": "user", "content": "hi"}],
        task,
    )
    assert prompt == v41.encode_messages(
        [{"role": "system", "content": "Rules."}, {"role": "user", "content": "hi"}],
        thinking_mode="chat",
    )
    assert "Reasoning Effort" not in prompt


def test_parser_extracts_official_v41_tool_calls_from_a_token_stream() -> None:
    calls = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Graz", "days": 3}),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": json.dumps({"city": "Wien <Innere Stadt>"}),
            },
        },
    ]
    block = _official_tool_call_block(calls)
    results = list(parse_deepseek_v41(_simulate(["Let me check.", *_chunks(block, 3)])))

    tool_results = [r for r in results if isinstance(r, ToolCallResponse)]
    text = "".join(r.text for r in results if isinstance(r, GenerationResponse))
    assert text == "Let me check."
    assert len(tool_results) == 1
    parsed = [
        (call.name, cast(dict[str, Any], json.loads(call.arguments)))
        for call in tool_results[0].tool_calls
    ]
    assert parsed == [
        ("get_weather", {"city": "Graz", "days": 3}),
        ("get_weather", {"city": "Wien <Innere Stadt>"}),
    ]


def test_dsml_body_parser_still_reads_unspaced_v4_tags() -> None:
    v4_block = (
        f'<{_DSML}invoke name="f">\n'
        f'<{_DSML}parameter name="x" string="false">1</{_DSML}parameter>\n'
        f"</{_DSML}invoke>"
    )
    parsed = parse_dsml_output(v4_block)
    assert parsed is not None
    assert [(c.name, c.arguments) for c in parsed] == [("f", '{"x": 1}')]


def test_v4_parser_does_not_claim_v41_blocks() -> None:
    block = _official_tool_call_block(
        [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]
    )
    results = list(parse_deepseek_v4(_simulate(_chunks(block, 4))))
    assert not any(isinstance(r, ToolCallResponse) for r in results)
