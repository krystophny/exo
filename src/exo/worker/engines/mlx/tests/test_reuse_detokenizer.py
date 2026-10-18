"""A tokenizer builds its streaming detokenizer once, and every request still gets its own."""

from typing import cast

import pytest
from mlx_lm.tokenizer_utils import (
    BPEStreamingDetokenizer,
    StreamingDetokenizer,
    TokenizerWrapper,
)

from exo.worker.engines.mlx.patches.reuse_detokenizer import (
    _fresh_detokenizer,  # pyright: ignore[reportPrivateUsage]
)


class CountingTokenizer:
    """Stands in for a Hugging Face tokenizer, counting how often its vocabulary is read."""

    clean_up_tokenization_spaces = False

    def __init__(self) -> None:
        self.vocab_reads = 0

    _VOCAB: dict[str, int] = {"Hello": 0, "Ġthere": 1, "Ġfriend": 2}

    @property
    def vocab(self) -> dict[str, int]:
        self.vocab_reads += 1
        return dict(self._VOCAB)

    # mlx-lm versions whose BPE detokenizer builds its token map by id read the
    # vocabulary through these instead
    def __len__(self) -> int:
        return len(self._VOCAB)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        self.vocab_reads += 1
        by_id = {token_id: token for token, token_id in self._VOCAB.items()}
        return [by_id[token_id] for token_id in ids]


def wrap(hf_tokenizer: CountingTokenizer) -> TokenizerWrapper:
    # Only the detokenizer is used here: skip the wrapper's set-up, which reads much more
    # of the tokenizer
    tokenizer = TokenizerWrapper.__new__(TokenizerWrapper)
    tokenizer.__dict__.update(
        _tokenizer=hf_tokenizer, _detokenizer_class=BPEStreamingDetokenizer
    )
    # mlx-lm versions that already build the detokenizer once keep it here, set by
    # TokenizerWrapper.__init__, and copy it per caller.
    if "_detokenizer" in TokenizerWrapper.__init__.__code__.co_names:
        tokenizer.__dict__["_detokenizer"] = BPEStreamingDetokenizer(hf_tokenizer)
    return tokenizer


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TokenizerWrapper, "detokenizer", property(_fresh_detokenizer))


@pytest.mark.usefixtures("patched")
def test_the_vocabulary_is_walked_once_per_tokenizer() -> None:
    hf_tokenizer = CountingTokenizer()
    tokenizer = wrap(hf_tokenizer)

    _ = tokenizer.detokenizer
    reads_for_one = hf_tokenizer.vocab_reads
    for _ in range(5):
        _ = tokenizer.detokenizer

    assert reads_for_one > 0
    assert hf_tokenizer.vocab_reads == reads_for_one


@pytest.mark.usefixtures("patched")
def test_each_request_gets_its_own_detokenizer_state() -> None:
    tokenizer = wrap(CountingTokenizer())

    def decode(detokenizer: StreamingDetokenizer, tokens: list[int]) -> str:
        text = ""
        for token in tokens:
            detokenizer.add_token(token)
            text += detokenizer.last_segment
        detokenizer.finalize()
        return text + detokenizer.last_segment

    first = cast(StreamingDetokenizer, tokenizer.detokenizer)
    first.add_token(0)
    # A second request starts while the first is under way
    second = cast(StreamingDetokenizer, tokenizer.detokenizer)

    assert decode(second, [0, 2]) == "Hello friend"
    assert first.last_segment + decode(first, [1]) == "Hello there"
