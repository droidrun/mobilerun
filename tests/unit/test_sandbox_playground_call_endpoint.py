import ast
import base64
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.support import sandbox_playground

_CONNECTION_OVERRIDES = re.compile(
    r"https?://\S+"
    r"|\bBearer\s+(?!<)[\w.-]+"
    r"|\b(?:MOBILERUN_)?(?:API_KEY|BASE_URL)\s*[:=]\s*['\"]?\S+"
    r"|\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@pytest.fixture
def caller(monkeypatch):
    from tests.support.sandbox_playground import call_endpoint

    monkeypatch.setattr(call_endpoint, "time", SimpleNamespace(monotonic=lambda: 0.0))
    monkeypatch.setenv("MOBILERUN_API_KEY", "fake-test-credential")
    return call_endpoint


def test_nested_long_base64_is_redacted(caller):
    encoded = base64.b64encode(b"abc" * 1000).decode()
    assert caller.redact({"nested": [encoded]})["nested"][0] == "<base64 jpeg, 3 KB>"


def test_pretty_print_and_save_screenshot(caller, tmp_path, capsys):
    image = b"jpeg" * 400
    screenshot = base64.b64encode(image).decode()
    cloud = MagicMock()
    run = cloud["demo"].fire.return_value
    run.invocation_id = "inv"
    run.id = "exec"
    run.result.return_value = {"message": "ok", "screenshot": screenshot}

    assert caller.main("demo", output_dir=tmp_path, endpoints=cloud) == 0
    assert (tmp_path / "demo.last.jpg").read_bytes() == image
    out = capsys.readouterr().out
    assert "<base64 jpeg, 2 KB>" in out
    assert screenshot not in out
    assert "fired invocation inv" in out
    assert "execution exec running" in out
    cloud["demo"].fire.assert_called_once_with()


def test_missing_key(caller, tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("MOBILERUN_API_KEY")
    cloud = MagicMock()
    assert caller.main("demo", output_dir=tmp_path, endpoints=cloud) == 1
    assert "run `source ~/.agent-env` first" in capsys.readouterr().err
    cloud.__getitem__.assert_not_called()


@pytest.mark.parametrize("name", ["../escape", "/absolute", r"..\escape", "..", ""])
def test_rejects_path_traversal_before_firing(caller, name, tmp_path, capsys):
    cloud = MagicMock()
    output_dir = tmp_path / "images"
    assert caller.main(name, output_dir=output_dir, endpoints=cloud) == 1
    assert "Invalid endpoint name" in capsys.readouterr().err
    assert not output_dir.exists()
    assert not (tmp_path / "escape.last.jpg").exists()
    cloud.__getitem__.assert_not_called()


def test_refuses_existing_symlink(caller, tmp_path, capsys):
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"unchanged")
    output_dir = tmp_path / "images"
    output_dir.mkdir()
    (output_dir / "demo.last.jpg").symlink_to(outside)
    cloud = MagicMock()
    assert caller.main("demo", output_dir=output_dir, endpoints=cloud) == 1
    assert "Invalid screenshot path" in capsys.readouterr().err
    assert outside.read_bytes() == b"unchanged"
    cloud.__getitem__.assert_not_called()


@pytest.mark.parametrize(
    "unsafe",
    [
        "https://example.test/private",
        "Bearer fake-credential",
        "API_KEY='fake-credential'",
        "01234567-89ab-cdef-0123-456789abcdef",
    ],
)
def test_generic_connection_hygiene_patterns(unsafe):
    assert _CONNECTION_OVERRIDES.search(unsafe)


def test_playground_fixtures_have_no_credentials_urls_or_device_ids():
    root = Path(sandbox_playground.__file__).parent
    forbidden = _CONNECTION_OVERRIDES
    for path in (*sorted(root.glob("*.py")), *sorted(root.glob("*.json"))):
        source = path.read_text(encoding="utf-8")
        assert not forbidden.search(path.name), path.name
        assert not forbidden.search(source), path.name
        if path.suffix == ".py":
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.keyword) and node.arg in {
                    "device",
                    "api_key",
                    "base_url",
                }:
                    assert (
                        isinstance(node.value, ast.Constant)
                        and node.value.value is None
                    ), path.name
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant):
                            assert key.value not in {
                                "MOBILERUN_API_KEY",
                                "MOBILERUN_BASE_URL",
                                "MOBILERUN_CLOUD_BASE_URL",
                            }, path.name
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not forbidden.search(node.value), path.name
