"""Workspace file transport behavior without cloud calls."""

import asyncio
import logging
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mobilerun_sdk import AsyncMobilerun

from mobilerun.errors import CloudPermissionDenied
from mobilerun.models import Billing


def test_upload_confirms_only_after_put_and_cancels_on_failure(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import files

    path = tmp_path / "photo.jpg"
    path.write_bytes(b"image")
    now = datetime.now(timezone.utc)
    resource = SimpleNamespace(
        upload_url=AsyncMock(
            return_value=SimpleNamespace(file_id="f1", put_url="https://put")
        ),
        confirm=AsyncMock(
            return_value=SimpleNamespace(
                file_id="f1",
                filename=path.name,
                mime_type="image/jpeg",
                size_bytes=5,
                created_at=now,
            )
        ),
        cancel_pending=AsyncMock(),
    )
    client = SimpleNamespace(files=resource)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(files.httpx, "AsyncClient", lambda: http)
    result = asyncio.run(files.upload_file(client, path, billing=Billing.NO_COMMERCE))
    assert (result.id, result.name, result.size_bytes) == ("f1", "photo.jpg", 5)
    resource.upload_url.assert_awaited_once_with(
        filename="photo.jpg", mime_type="image/jpeg", size_bytes=5, zone="user"
    )
    http.put.assert_awaited_once_with(
        "https://put",
        content=b"image",
        headers={"Content-Type": "image/jpeg"},
        timeout=None,
    )
    resource.confirm.assert_awaited_once_with("f1")
    http.put.side_effect = RuntimeError("failure")
    with pytest.raises(RuntimeError, match="failure"):
        asyncio.run(files.upload_file(client, path, billing=Billing.NO_COMMERCE))
    resource.cancel_pending.assert_awaited_once_with("f1")


def test_list_download_and_delete(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import files

    now = datetime.now(timezone.utc)
    item = SimpleNamespace(
        file_id="f1",
        filename="photo.jpg",
        mime_type="image/jpeg",
        size_bytes=5,
        created_at=now,
    )
    raw = SimpleNamespace(read=AsyncMock(return_value=b"image"))
    resource = SimpleNamespace(
        list=AsyncMock(return_value=SimpleNamespace(files=[item])),
        delete=AsyncMock(),
        with_raw_response=SimpleNamespace(download=AsyncMock(return_value=raw)),
    )
    client = SimpleNamespace(files=resource)
    items = asyncio.run(files.list_files(client, billing=Billing.NO_COMMERCE))
    assert items[0].name == "photo.jpg"
    assert (
        asyncio.run(files.download_file(client, "f1", billing=Billing.NO_COMMERCE))
        == b"image"
    )
    asyncio.run(files.delete_file(client, "f1", billing=Billing.NO_COMMERCE))
    resource.list.assert_awaited_once_with(zone="user")
    resource.with_raw_response.download.assert_awaited_once_with("f1")
    resource.delete.assert_awaited_once_with("f1")


def test_device_file_sdk_transport(tmp_path):
    from mobilerun.clients.cloud import files

    path = tmp_path / "picture.png"
    path.write_bytes(b"png")
    item = SimpleNamespace(name="picture.png", size=3, modified_at=None)
    raw = SimpleNamespace(read=AsyncMock(return_value=b"png"))
    resource = SimpleNamespace(
        upload=AsyncMock(),
        list=AsyncMock(return_value=SimpleNamespace(files=[item])),
        delete=AsyncMock(),
        with_raw_response=SimpleNamespace(download=AsyncMock(return_value=raw)),
    )
    client = SimpleNamespace(devices=SimpleNamespace(files=resource))
    asyncio.run(
        files.push_device_file(
            client, "device", path, "/sdcard", billing=Billing.NO_COMMERCE
        )
    )
    resource.upload.assert_awaited_once_with("device", path="/sdcard", file=path)
    listed = asyncio.run(
        files.list_device_files(
            client, "device", "/sdcard", billing=Billing.NO_COMMERCE
        )
    )
    assert listed[0].path == "/sdcard/picture.png"
    assert (
        asyncio.run(
            files.download_device_file(
                client, "device", "/sdcard/picture.png", billing=Billing.NO_COMMERCE
            )
        )
        == b"png"
    )
    asyncio.run(
        files.delete_device_file(
            client, "device", "/sdcard/picture.png", billing=Billing.NO_COMMERCE
        )
    )
    resource.delete.assert_awaited_once_with("device", path="/sdcard/picture.png")


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("status", [200, 403])
def test_sdk_download_redirect_hides_signature_and_strips_auth(caplog, device, status):
    from mobilerun.clients.cloud import files

    requests = []
    signature = "storage-redirect-canary"
    caplog.set_level(logging.INFO, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="mobilerun_sdk._base_client")

    def respond(request):
        requests.append(request)
        if request.url.host == "api.example":
            return httpx.Response(
                302,
                headers={"Location": f"https://storage.example/signed?sig={signature}"},
            )
        return httpx.Response(
            status, content=b"\x00\xff" if status == 200 else b"denied"
        )

    async def run():
        http = httpx.AsyncClient(
            transport=httpx.MockTransport(respond), follow_redirects=True
        )
        async with AsyncMobilerun(
            api_key="test", base_url="https://api.example", http_client=http
        ) as client:
            if device:
                return await files.download_device_file(
                    client, "id", "/sdcard/test", billing=Billing.NO_COMMERCE
                )
            return await files.download_file(client, "id", billing=Billing.NO_COMMERCE)

    if status == 200:
        assert asyncio.run(run()) == b"\x00\xff"
    else:
        with pytest.raises(CloudPermissionDenied) as caught:
            asyncio.run(run())
        assert signature not in "".join(traceback.format_exception(caught.value))
    assert len(requests) == 2
    assert "authorization" not in requests[1].headers
    assert requests[1].url.params["sig"] == signature
    assert signature not in caplog.text
