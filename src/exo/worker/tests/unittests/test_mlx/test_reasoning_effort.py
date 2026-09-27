from typing import cast
from unittest.mock import Mock

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams
from exo.worker.engines.mlx.utils_mlx import render_chat_template


def test_template_preserves_reasoning_and_model_options() -> None:
    apply_template = Mock(return_value="prompt")
    tokenizer = Mock(spec=TokenizerWrapper, apply_chat_template=apply_template)
    result = render_chat_template(
        cast(TokenizerWrapper, tokenizer),
        [{"role": "user", "content": "hi"}],
        TextGenerationTaskParams(
            model=ModelId("pipenetwork/GLM-5.3-MLX-mixed-4_8bit"),
            input=[],
            reasoning_effort="max",
            chat_template_kwargs={"clear_thinking": True},
        ),
    )
    assert result == "prompt"
    apply_template.assert_called_once_with(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        reasoning_effort="max",
        clear_thinking=True,
    )
