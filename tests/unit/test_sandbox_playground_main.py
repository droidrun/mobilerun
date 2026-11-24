from types import SimpleNamespace
from unittest.mock import Mock, patch

from mobilerun_functions import RunInfo

from tests.support.sandbox_playground import main as snapshot


def test_collects_device_snapshot_and_opens_requested_app(tmp_path):
    shot = tmp_path / "screen.png"
    shot.write_bytes(b"png")
    device = Mock(device_id="fake-device")
    device.screenshot.return_value = str(shot)
    device.ui.return_value = SimpleNamespace(
        nodes=[{"text": "Hello"}, {"text": ""}, {"text": "World"}]
    )
    log = Mock()
    ctx = SimpleNamespace(
        device=lambda: device, log=log, run=RunInfo(device_id=device.device_id)
    )

    with patch("time.monotonic", return_value=0):
        result = snapshot.main(snapshot.Input(app_package="example.app"), ctx)

    assert result.device_id == "fake-device"
    assert result.screenshot_path == str(shot)
    assert result.ui_node_count == 3
    assert result.top_texts == ["Hello", "World"]
    device.start_app.assert_called_once_with("example.app")
    logged = " ".join(call.args[0] for call in log.info.call_args_list)
    assert "Opening app example.app" in logged
    assert "3 nodes, 2 with text" in logged
    log.warning.assert_not_called()
