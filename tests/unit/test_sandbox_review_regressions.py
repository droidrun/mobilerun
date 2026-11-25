"""Regression tests for owned schedules, repeat deployments, and wire results."""

import hashlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from mobilerun.sandbox import Remote
from mobilerun.sandbox import deployment as d
from mobilerun.sandbox.errors import SandboxDeclarationError, SandboxDeploymentConflict
from mobilerun.sandbox.models import CloudDeclaration
from tests.unit.test_sandbox_redeploy import changed, setup_deploy
from tests.unit.test_sandbox_remote import Input, backend

__all__ = ["changed", "setup_deploy", "backend"]


@pytest.mark.anyio
async def test_disable_without_document(changed):
    changed.cloud.fail = "read_document"
    assert not (await d.set_enabled("demo", False))["enabled"]
    assert changed.cloud.calls == ["disable"]


@pytest.mark.anyio
async def test_rollback_then_identical_deploy(changed, monkeypatch):
    cloud = changed.cloud
    keys, sources = {}, {1: "source"}

    async def upload(fid, source, key, **kwargs):
        if key not in keys:
            cloud.version += 1
            sources[cloud.version] = source
            keys[key] = cloud.version
        return {"version": keys[key]}

    async def get_source(fid, version, **kwargs):
        return sources[version]

    monkeypatch.setattr(cloud, "create_version", upload)
    monkeypatch.setattr(cloud, "get_source", get_source)
    for src in ["B", "C", "B", "B"]:
        cloud.calls.clear()
        result = await d.deploy_file(
            Path("demo.py"),
            artifact=replace(changed.artifact, source=src, digest=src),
            yes=True,
        )
    assert result.status == "unchanged"
    assert result.version == 4 and cloud.calls == ["read_document"]
    # Another caller may have pushed an unpinned version since the last deploy.
    cloud.version += 1
    sources[cloud.version] = "other"
    cloud.calls.clear()
    result = await d.deploy_file(
        Path("demo.py"),
        artifact=replace(changed.artifact, source="B", digest="B"),
        yes=True,
    )
    assert result.version == 4 and cloud.calls == ["read_document"]


def test_url_aliases_recover_ownership_or_fail_closed(tmp_path, monkeypatch):
    base = "https://example.test/v1"
    monkeypatch.setattr(d, "resolve_cloud_api_key", lambda: "key")
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path))

    def scoped(url):
        monkeypatch.setattr(
            d, "get_settings", lambda: SimpleNamespace(MOBILERUN_CLOUD_BASE_URL=url)
        )
        return d.current_scope()

    assert scoped(base) == scoped(base + "/")
    for url, fid in [(base, "f"), (base + "/", "other")]:
        account = scoped(url).account_id
        digest = hashlib.sha256((url + "\0" + account).encode()).hexdigest()
        directory = tmp_path / digest / hashlib.sha256(b"demo").hexdigest()
        directory.mkdir(parents=True)
        receipt = d.OwnershipReceipt(
            url, account, "demo", fid, None, None, (), None, ()
        )
        (directory / "receipt.json").write_text(json.dumps(receipt.to_dict()))
        if fid == "f":
            assert d.ReceiptStore(scoped(base + "/"), "demo").read().function_id == "f"
    with pytest.raises(SandboxDeploymentConflict):
        d.ReceiptStore(scoped(base), "demo").read()


@pytest.mark.anyio
async def test_schedule_rejects_required_input_before_mutation(changed):
    art = replace(
        changed.artifact,
        declaration=CloudDeclaration("demo", schedule="* * * * *"),
        source="from mobilerun_functions import script\nfrom pydantic import BaseModel\nclass Required(BaseModel):\n    count: int\n@script\ndef task(input: Required) -> Required: return input\n",
        input_schema={"type": "object"},  # No model-inspection evidence.
    )
    with pytest.raises(SandboxDeclarationError, match="count"):
        await d.deploy_file(Path("demo.py"), artifact=art, device_free=True, yes=True)
    assert not changed.cloud.calls


def test_date_result_decodes_in_json_mode(backend):
    class Out(BaseModel):
        model_config = ConfigDict(strict=True)
        day: date

    backend["wait_result"].return_value["result"]["steps"][0]["body"]["result"] = {
        "day": "2026-01-01"
    }
    assert Remote("demo", Input, Out).fire(n=1).result().day == date(2026, 1, 1)
