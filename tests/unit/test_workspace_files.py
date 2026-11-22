import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from mobilerun.models import Billing, WorkspaceFile


def test_workspace_api_owns_client_and_downloads_into_directory(monkeypatch, tmp_path):
    from mobilerun.core import workspace as workspace_module
    from mobilerun.core.workspace import Workspace as files

    item = WorkspaceFile("id", "photo.jpg", "image/jpeg", 5, datetime.now(timezone.utc))
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(
        workspace_module, "cloud_client", lambda *args, **kwargs: client
    )
    monkeypatch.setattr(
        workspace_module.cloud_files, "upload_file", AsyncMock(return_value=item)
    )
    monkeypatch.setattr(
        workspace_module.cloud_files, "list_files", AsyncMock(return_value=[item])
    )
    monkeypatch.setattr(
        workspace_module.cloud_files, "download_file", AsyncMock(return_value=b"image")
    )
    monkeypatch.setattr(workspace_module.cloud_files, "delete_file", AsyncMock())

    assert (
        asyncio.run(files.upload_file("photo.jpg", billing=Billing.NO_COMMERCE)) == item
    )
    assert asyncio.run(files.list_files(billing=Billing.NO_COMMERCE)) == [item]
    result = asyncio.run(
        files.download_file("id", tmp_path, billing=Billing.NO_COMMERCE)
    )
    assert result == tmp_path / "photo.jpg"
    assert result.read_bytes() == b"image"
    asyncio.run(files.delete_file("id", billing=Billing.NO_COMMERCE))
    workspace_module.cloud_files.delete_file.assert_awaited_once_with(
        client, "id", billing=Billing.NO_COMMERCE
    )


def test_download_rejects_unknown_file(monkeypatch, tmp_path):
    from mobilerun.core import workspace as workspace_module
    from mobilerun.core.workspace import Workspace as files

    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(
        workspace_module, "cloud_client", lambda *args, **kwargs: client
    )
    monkeypatch.setattr(
        workspace_module.cloud_files, "list_files", AsyncMock(return_value=[])
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            files.download_file("missing", tmp_path, billing=Billing.NO_COMMERCE)
        )
