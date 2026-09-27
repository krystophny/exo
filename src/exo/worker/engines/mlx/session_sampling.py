"""Native sampling filters with request-local random keys for interleaving."""

from collections.abc import Callable

import mlx.core as mx
from mlx_lm.sample_utils import apply_min_p, apply_top_k, apply_top_p


def make_session_sampler(
    seed: int, temperature: float, top_p: float, min_p: float, top_k: int
) -> Callable[[mx.array], mx.array]:
    key = mx.random.key(seed)

    def sample(logprobs: mx.array) -> mx.array:
        nonlocal key
        if temperature == 0:
            return mx.argmax(logprobs, axis=-1)
        if 0 < top_p < 1:
            logprobs = apply_top_p(logprobs, top_p)
        if min_p != 0:
            logprobs = apply_min_p(logprobs, min_p, 1)
        if top_k > 0:
            logprobs = apply_top_k(logprobs, top_k)
        keys = mx.random.split(key)
        key = keys[0]
        return mx.random.categorical(logprobs / temperature, key=keys[1])

    return sample
