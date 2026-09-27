"""Native MTP configuration for a single text session."""

import os
from pathlib import Path

from mlx import nn
from mlx_lm.models.deepseek_v32 import Model as DeepseekV32Model


def mtp_enabled(model: nn.Module) -> bool:
    if os.getenv("EXO_MTP", "0") != "1":
        return False
    if not isinstance(model, DeepseekV32Model) or not model.has_mtp:
        raise ValueError("EXO_MTP requires a loaded native DeepSeek/GLM MTP head")
    if (
        os.getenv("EXO_NO_BATCH") != "1"
        or os.getenv("EXO_PREFIX_CACHE_SINGLE_SESSION") != "1"
    ):
        raise ValueError(
            "Native MTP currently requires unbatched single-session caching"
        )
    return True


def mtp_weights_path() -> Path | None:
    path = os.getenv("EXO_MTP_WEIGHTS_DIR")
    return Path(path).expanduser() if os.getenv("EXO_MTP") == "1" and path else None
