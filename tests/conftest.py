"""Shared fixtures and marker handling. gpu / api / webcam markers skip when the resource is absent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _has_gpu() -> bool:
    try:
        import torch  # noqa: F401  (heavy import; only when a gpu test is collected)

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    skip_api = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    skip_cam = pytest.mark.skip(reason="set ALLY_WEBCAM=1 to run webcam tests")
    gpu_ok: bool | None = None
    for item in items:
        if "gpu" in item.keywords:
            gpu_ok = _has_gpu() if gpu_ok is None else gpu_ok
            if not gpu_ok:
                item.add_marker(skip_gpu)
        if "api" in item.keywords and not os.environ.get("ANTHROPIC_API_KEY"):
            item.add_marker(skip_api)
        if "webcam" in item.keywords and os.environ.get("ALLY_WEBCAM") != "1":
            item.add_marker(skip_cam)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings built from defaults only: no .env file, no ALLY_* / secret variables from the shell."""
    for key in list(os.environ):
        if key.startswith(("ALLY_", "ANTHROPIC_", "TELEGRAM_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(REPO_ROOT / "tests")  # no .env here
