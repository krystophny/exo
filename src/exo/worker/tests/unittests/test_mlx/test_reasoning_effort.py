from typing import Any

from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.utils_mlx import render_chat_template

_MESSAGES: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]


class _Hy3Tokenizer:
    """Enforces Hy3's chat-template contract: any reasoning_effort other than
    no_think/low/high raises, exactly as the real template does before it
    crashes the runner."""

    chat_template = "stub"

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.seen = kwargs
        effort = kwargs.get("reasoning_effort")
        if effort is not None and effort not in ("no_think", "low", "high"):
            raise ValueError(f"reasoning_effort error : {effort}")
        return "PROMPT"


class _RecordingTokenizer:
    chat_template = "stub"

    def __init__(self) -> None:
        self.seen: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.seen = kwargs
        return "PROMPT"


def _params(model: str, **kw: Any) -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=model,
        input=[InputMessage(role="user", content="hi")],
        **kw,
    )


def test_hy3_medium_effort_maps_to_low() -> None:
    tok = _Hy3Tokenizer()
    prompt = render_chat_template(
        tok,
        list(_MESSAGES),
        _params("tencent/Hy3-preview", enable_thinking=True, reasoning_effort="medium"),
    )
    assert prompt == "PROMPT"
    assert tok.seen["reasoning_effort"] == "low"


def test_hy3_high_effort_maps_to_high() -> None:
    tok = _Hy3Tokenizer()
    render_chat_template(
        tok, list(_MESSAGES), _params("kernelpool/Hy3-6bit", reasoning_effort="xhigh")
    )
    assert tok.seen["reasoning_effort"] == "high"


def test_hy3_max_effort_maps_to_high() -> None:
    tok = _Hy3Tokenizer()
    render_chat_template(
        tok, list(_MESSAGES), _params("kernelpool/Hy3-6bit", reasoning_effort="max")
    )
    assert tok.seen["reasoning_effort"] == "high"


def test_hy3_thinking_disabled_maps_to_no_think() -> None:
    tok = _Hy3Tokenizer()
    render_chat_template(
        tok, list(_MESSAGES), _params("tencent/Hy3", enable_thinking=False)
    )
    assert tok.seen["reasoning_effort"] == "no_think"


def test_non_hy3_effort_passes_through_unchanged() -> None:
    tok = _RecordingTokenizer()
    render_chat_template(
        tok,
        list(_MESSAGES),
        _params("mlx-community/GLM-5.2-mxfp4", reasoning_effort="medium"),
    )
    assert tok.seen["reasoning_effort"] == "medium"


def test_glm_max_effort_passes_through_unchanged() -> None:
    tok = _RecordingTokenizer()
    render_chat_template(
        tok,
        list(_MESSAGES),
        _params("mlx-community/GLM-5.2-mxfp4", reasoning_effort="max"),
    )
    assert tok.seen["reasoning_effort"] == "max"
