from exo.shared.models.model_cards import ConfigData


def test_hy3_config_supports_tensor_parallel() -> None:
    config_data = ConfigData.model_validate(
        {
            "architectures": ["HYV3ForCausalLM"],
            "hidden_size": 4096,
            "num_key_value_heads": 8,
            "num_hidden_layers": 80,
            "max_position_embeddings": 262144,
        }
    )

    assert config_data.supports_tensor is True
