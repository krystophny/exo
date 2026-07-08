import pytest

from exo.worker.engines.mlx.constants import mlx_prefill_step_size


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"EXO_PREFILL_STEP_SIZE": "2048"}, 2048),
        ({"EXO_MLX_PREFILL_STEP_SIZE": "1024"}, 1024),
        (
            {
                "EXO_PREFILL_STEP_SIZE": "512",
                "EXO_MLX_PREFILL_STEP_SIZE": "1024",
            },
            512,
        ),
        (
            {
                "EXO_PREFILL_STEP_SIZE": "not-an-int",
                "EXO_MLX_PREFILL_STEP_SIZE": "1024",
            },
            1024,
        ),
        (
            {
                "EXO_PREFILL_STEP_SIZE": "0",
                "EXO_MLX_PREFILL_STEP_SIZE": "-1",
            },
            4096,
        ),
    ],
)
def test_mlx_prefill_step_size_env_names(
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    expected: int,
) -> None:
    monkeypatch.delenv("EXO_PREFILL_STEP_SIZE", raising=False)
    monkeypatch.delenv("EXO_MLX_PREFILL_STEP_SIZE", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert mlx_prefill_step_size() == expected
