from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest


def test_file_models_are_immutable_value_objects() -> None:
    from mobilerun.models.files import DeviceFile, WorkspaceFile

    now = datetime.now(timezone.utc)
    workspace = WorkspaceFile("id", "photo.jpg", "image/jpeg", 12, now)
    device = DeviceFile("/sdcard/photo.jpg", "photo.jpg", 12, now)
    assert workspace.name == device.name
    assert workspace.size_bytes == device.size_bytes
    with pytest.raises(FrozenInstanceError):
        workspace.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        device.path = "changed"  # type: ignore[misc]
