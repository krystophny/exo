import io
from pathlib import Path

import pytest

from exo.utils import banner


@pytest.fixture
def marker(tmp_path: Path) -> Path:
    return tmp_path / ".dashboard_opened"


@pytest.fixture
def first_run(marker: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []
    monkeypatch.setattr(banner, "_FIRST_RUN_MARKER", marker)
    monkeypatch.setattr(banner.webbrowser, "open", opened.append)
    monkeypatch.delenv("EXO_RUNTIME_DIR", raising=False)
    return opened


class _Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_headless_first_run_never_opens_a_browser(
    first_run: list[str], marker: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(banner.sys, "stderr", io.StringIO())
    banner.print_startup_banner(52415)
    assert first_run == []
    assert marker.exists()


def test_interactive_first_run_opens_the_dashboard(
    first_run: list[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(banner.sys, "stderr", _Tty())
    banner.print_startup_banner(52415)
    assert first_run == ["http://localhost:52415"]
