import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from mobilerun.errors import CloudConnectionError


def test_upload_confirms_and_marks_failure(monkeypatch, tmp_path):
    from mobilerun.clients.cloud import apps

    apk = tmp_path / "sample.apk"
    apk.write_bytes(b"apk")
    monkeypatch.setattr(
        apps, "inspect_apk", lambda path: ("org.example", 3, "1.2", "Example")
    )
    signed = SimpleNamespace(
        app_id="app-id",
        version_id="version-id",
        r2_upload_urls=[
            SimpleNamespace(file_name="sample.apk", r2_upload_url="https://upload")
        ],
    )
    sdk = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                return_value=SimpleNamespace(
                    items=[], pagination=SimpleNamespace(has_next=False)
                )
            ),
            create_signed_upload_url=AsyncMock(return_value=signed),
            confirm_upload=AsyncMock(),
            mark_failed=AsyncMock(),
        )
    )
    client = AsyncMock()
    client.__aenter__.return_value = sdk
    monkeypatch.setattr(apps, "cloud_client", lambda *args, **kwargs: client)
    http = AsyncMock()
    http.__aenter__.return_value = http
    http.put.return_value = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr(apps.httpx, "AsyncClient", lambda **kwargs: http)
    result = asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    assert result.id == "app-id"
    sdk.apps.confirm_upload.assert_awaited_once_with("version-id")
    http.put.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    sdk.apps.mark_failed.assert_awaited_once_with("version-id")
    sdk.apps.mark_failed.reset_mock()
    http.put.side_effect = httpx.ConnectError("https://signed.test?token=private")
    with pytest.raises(CloudConnectionError) as raised:
        asyncio.run(apps.upload_app(apk, billing="included", api_key="test"))
    assert "signed.test" not in str(raised.value)
    assert "private" not in str(raised.value)
    assert raised.value.__cause__ is http.put.side_effect
    sdk.apps.mark_failed.assert_awaited_once_with("version-id")


def test_app_items_paginates():
    from mobilerun.clients.cloud import apps

    item = SimpleNamespace(id="id")
    client = SimpleNamespace(
        apps=SimpleNamespace(
            list=AsyncMock(
                side_effect=[
                    SimpleNamespace(
                        items=[], pagination=SimpleNamespace(has_next=True)
                    ),
                    SimpleNamespace(
                        items=[item], pagination=SimpleNamespace(has_next=False)
                    ),
                ]
            )
        )
    )

    async def collect():
        return [value async for value in apps._items(client)]

    assert asyncio.run(collect()) == [item]
    assert client.apps.list.await_count == 2


def test_apk_metadata_from_build_tools(monkeypatch, tmp_path):
    from mobilerun.clients.local import apk as apps

    monkeypatch.setattr(apps.shutil, "which", lambda name: "/sdk/aapt")
    monkeypatch.setattr(
        apps.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="package: name='org.example' versionCode='12' versionName='1.2'\napplication-label:'Example'\n"
        ),
    )
    assert apps.inspect_apk(tmp_path / "app.apk") == (
        "org.example",
        12,
        "1.2",
        "Example",
    )
