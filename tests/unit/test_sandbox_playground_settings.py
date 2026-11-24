import base64
import io
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from mobilerun.sandbox.deployment import build_document
from mobilerun.sandbox.models import Artifact, CloudDeclaration
from tests.support.sandbox_playground import settings as script


def run_settings(ctx):
    with patch.multiple("time", monotonic=lambda: 0.0, sleep=Mock()):
        return script.main(script.Input(wait_s=0), ctx)


@pytest.fixture
def run_device(tmp_path):
    def run(app="com.android.settings"):
        path = tmp_path / "screen.png"
        Image.new("RGB", (1200, 1800), "blue").save(path)
        device = Mock()
        device.screenshot.return_value = str(path)
        device.current_app_id.return_value = app
        device.ui.return_value = SimpleNamespace(
            nodes=[{"text": "Settings"}, {"text": ""}, {"text": "Network"}]
        )
        ctx = SimpleNamespace(device=lambda: device, log=Mock(), partial=Mock())
        return run_settings(ctx), device, ctx

    return run


def test_success_and_home_cleanup(run_device):
    result, device, ctx = run_device()
    assert result.settings_open
    assert result.texts == ["Settings", "Network"]
    assert device.key.call_count == 2
    assert device.key.call_args_list[0].args == ("home",)
    assert device.key.call_args_list[-1].args == ("home",)
    device.start_app.assert_called_once_with("com.android.settings")
    with Image.open(io.BytesIO(base64.b64decode(result.screenshot))) as image:
        assert max(image.size) <= 1000
    assert len(result.screenshot) <= 240000
    ctx.partial.assert_not_called()


def test_wrong_app(run_device):
    result, _, _ = run_device(app="other")
    assert not result.settings_open


def test_budget_exceeded_marks_partial(run_device):
    with patch("base64.b64encode", return_value=b"x" * 300000):
        result, _, ctx = run_device()
    assert result.screenshot is None
    ctx.partial.assert_called_once()


def test_endpoint_document_binds_device():
    artifact = Artifact(CloudDeclaration("x"), "", "", {}, None)
    document = build_document(artifact, "fn", 1, device="fake-device")
    assert document["devices"] == ["fake-device"]
    assert "custom" in document["on"]


def test_home_after_failure():
    device = Mock()
    device.start_app.side_effect = RuntimeError("failed")
    ctx = SimpleNamespace(device=lambda: device, log=Mock())
    with pytest.raises(RuntimeError, match="failed"):
        run_settings(ctx)
    assert device.key.call_count == 2
    assert device.key.call_args.args == ("home",)
