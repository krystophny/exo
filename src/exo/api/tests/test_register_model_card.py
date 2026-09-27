# pyright: reportUnusedFunction=false, reportAny=false
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from exo.api.main import API
from exo.api.types.api import AddCustomModelParams
from exo.shared.models.model_cards import ModelCard, card_cache
from exo.shared.types.commands import ForwarderCommand
from exo.shared.types.common import ModelId, SystemId
from exo.utils.channels import Sender


@pytest.mark.parametrize("mismatch", [False, True])
async def test_register_supplied_card_preserves_offline_model_contract(
    mismatch: bool,
) -> None:
    card = ModelCard.model_validate(
        {
            "model_id": "test/offline-glm",
            "storage_size": {"in_bytes": 1024},
            "n_layers": 4,
            "hidden_size": 128,
            "supports_tensor": True,
            "tasks": ["TextGeneration"],
            "backends": ["MlxMetal"],
            "family": "glm",
            "capabilities": ["text", "thinking", "tools"],
            "reasoning_dialect": "post_last_user",
            "context_length": 4096,
            "trust_remote_code": False,
        }
    )
    sender = AsyncMock()
    api = object.__new__(API)
    api._system_id = SystemId()  # pyright: ignore[reportPrivateUsage]
    api.command_sender = cast(Sender[ForwarderCommand], sender)
    payload = AddCustomModelParams(
        model_id=ModelId("test/different") if mismatch else card.model_id,
        model_card=card,
    )
    if mismatch:
        with pytest.raises(HTTPException, match="Model card ID"):
            await api.add_custom_model(payload)
        sender.send.assert_not_called()
        return
    try:
        response = await api.add_custom_model(payload)
        assert response.id == card.model_id
        command = sender.send.call_args.args[0].command
        restored = ModelCard.model_validate_json(command.model_card.model_dump_json())
        assert restored.reasoning_dialect == "post_last_user"
        assert restored.capabilities == ["text", "thinking", "tools"]
        assert restored.context_length == 4096
        assert restored.is_custom
        assert card_cache.get(card.model_id) == restored
    finally:
        card_cache.cc.pop(card.model_id, None)
