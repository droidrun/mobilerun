"""Workspace catalog operations own their clients, not destination devices."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun import Workspace
from mobilerun.core import workspace
from mobilerun.models import App, Billing


def test_catalog_pagination_mapping_and_delete(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    first = SimpleNamespace(
        bundle_id="one",
        display_name="One",
        id="a",
        version=SimpleNamespace(version_name="1", version_code=3),
    )
    second = SimpleNamespace(
        bundle_id="two",
        display_name="Two",
        id="b",
        version=SimpleNamespace(version_name=None, version_code=4),
    )
    client.apps.list.side_effect = [
        SimpleNamespace(items=[first], pagination=SimpleNamespace(has_next=True)),
        SimpleNamespace(items=[second], pagination=SimpleNamespace(has_next=False)),
    ]
    calls = []

    def connect(api_key, base_url, *, billing):
        calls.append((api_key, base_url, billing))
        return client

    monkeypatch.setattr(workspace, "cloud_client", connect)
    options = dict(
        api_key="source", base_url="https://source.test", billing=Billing.NO_COMMERCE
    )
    assert asyncio.run(Workspace.list_apps(**options)) == [
        App("one", "1", "One", id="a", version_code=3),
        App("two", "", "Two", id="b", version_code=4),
    ]
    assert [call.kwargs["page"] for call in client.apps.list.await_args_list] == [1, 2]
    asyncio.run(Workspace.delete_app("a", **options))
    client.apps.delete.assert_awaited_once_with("a")
    assert calls == [("source", "https://source.test", Billing.NO_COMMERCE)] * 2
    assert client.__aexit__.await_count == 2


def test_get_app_default_billing_and_client_cleanup_on_failure(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.apps.retrieve.return_value = SimpleNamespace(
        data=SimpleNamespace(bundle_id="one", display_name="One")
    )
    calls = []

    def connect(api_key, base_url, *, billing):
        calls.append((api_key, base_url, billing))
        return client

    monkeypatch.setattr(workspace, "cloud_client", connect)
    assert asyncio.run(Workspace.get_app("a")) == App("one", "", "One", id="a")
    client.apps.retrieve.side_effect = RuntimeError("unavailable")
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(Workspace.get_app("b"))
    assert calls == [(None, None, Billing.NO_COMMERCE)] * 2
    assert client.__aexit__.await_count == 2


def test_workspace_upload_app_passes_overrides(monkeypatch):
    app = App("one", "1", "One", id="a")
    upload = AsyncMock(return_value=app)
    monkeypatch.setattr(workspace.cloud_apps, "upload_app", upload)
    assert (
        asyncio.run(
            Workspace.upload_app(
                "one.apk", api_key="key", base_url="https://source.test"
            )
        )
        is app
    )
    upload.assert_awaited_once_with(
        "one.apk",
        api_key="key",
        base_url="https://source.test",
        billing=Billing.NO_COMMERCE,
    )


def test_canonical_api_key_only_and_saved_login_fallback(monkeypatch):
    from mobilerun.clients.local import login_store

    monkeypatch.setattr(login_store, "load_credentials", lambda: {"api_key": "saved"})
    monkeypatch.setenv("MOBILERUN_CLOUD_API_KEY", "obsolete")
    monkeypatch.delenv("MOBILERUN_API_KEY", raising=False)
    assert login_store.resolve_cloud_api_key() == "saved"
    monkeypatch.setenv("MOBILERUN_API_KEY", "canonical")
    assert login_store.resolve_cloud_api_key() == "canonical"
