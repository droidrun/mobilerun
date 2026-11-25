"""Opt-in, potentially paid, device-free sandbox endpoint lifecycle."""

import asyncio
import os
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from textwrap import dedent
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.smoke_cloud]


def _verified_smoke_scope():
    """Refuse a default/ambiguous origin after pytest restores the saved environment."""
    origin = os.environ.get("MOBILERUN_CLOUD_BASE_URL")
    message = "Set an explicit unambiguous MOBILERUN_CLOUD_BASE_URL for the sandbox smoke; no default origin is permitted"
    if not origin or any(character.isspace() for character in origin):
        pytest.skip(message)
    try:
        parsed = urlsplit(origin)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname
            and not (
                parsed.username or parsed.password or parsed.query or parsed.fragment
            )
        )
        _ = parsed.port  # Validate a supplied port without contacting the origin.
    except ValueError:
        valid = False
    if not valid:
        pytest.skip(message)
    from mobilerun.sandbox.deployment import current_scope

    scope = current_scope()
    assert (
        scope.origin == origin or scope.origin == origin + "/"
    ), "Sandbox smoke origin differs from the restored explicit origin"
    return scope


def test_device_free_sandbox_endpoint(real_env, tmp_path, monkeypatch):
    _exercise_endpoint(tmp_path, monkeypatch, webhook=None)


def test_device_free_sandbox_webhook(real_env, tmp_path, monkeypatch):
    webhook = os.getenv("MOBILERUN_SMOKE_WEBHOOK_URL")
    if not webhook:
        pytest.skip(
            "Set MOBILERUN_SMOKE_WEBHOOK_URL to a dedicated HTTPS receiver you control"
        )
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Set MOBILERUN_API_KEY for the sandbox smoke")
    pytest.importorskip("mobilerun_functions")
    _verified_smoke_scope()
    from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT, CloudWebhooks
    from mobilerun.errors.cloud import CloudResourceNotFound
    from mobilerun.models.billing import Billing

    hooks = CloudWebhooks(billing=Billing.PAID)
    endpoint = asyncio.run(hooks.register(webhook, event_types=[DEFAULT_WEBHOOK_EVENT]))
    try:
        print(f"smoke webhook endpoint: {endpoint.id}", flush=True)
        _exercise_endpoint(tmp_path, monkeypatch, webhook=endpoint.id)
    finally:
        asyncio.run(hooks.delete(endpoint.id))
        with pytest.raises(CloudResourceNotFound):
            asyncio.run(hooks.get(endpoint.id))


def _exercise_endpoint(tmp_path, monkeypatch, webhook):
    with TemporaryDirectory(prefix="mobilerun-smoke-source-") as directory:
        _run_lifecycle(tmp_path, monkeypatch, webhook, Path(directory) / "ping.py")


def _run_lifecycle(tmp_path, monkeypatch, webhook, source):
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Set MOBILERUN_API_KEY for the sandbox smoke")
    pytest.importorskip("mobilerun_functions")
    pytest.importorskip("mobilerun_sdk")

    from mobilerun.clients.cloud.client import cloud_client
    from mobilerun.errors.cloud import CloudResourceNotFound
    from mobilerun.models.billing import Billing
    from mobilerun.sandbox import clients, deploy
    from mobilerun.sandbox.deployment import (
        ReceiptStore,
        cleanup_owned,
        current_scope,
    )
    from mobilerun.sandbox.errors import SandboxRunError

    scope = _verified_smoke_scope()
    name = f"sandbox-smoke-{uuid4().hex}"
    source.write_text(
        dedent(f"""\
            from pydantic import BaseModel
            from mobilerun.sandbox import cloud

            class Ping(BaseModel):
                message: str

            class Pong(BaseModel):
                reply: str

            @cloud({name!r}, webhook={webhook!r})
            def ping(input: Ping) -> Pong:
                return Pong(reply=f"pong: {{input.message}}")
            """),
        encoding="utf-8",
    )
    script = run_path(str(source))
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path / "sandbox"))
    store = ReceiptStore(scope, name)
    read_mode = Billing.NO_COMMERCE

    assert store.read() is None
    print(
        f"sandbox smoke: name={name} source={source} receipt={store.receipt_path}",
        flush=True,
    )

    async def verify_removed(receipt):
        if receipt.function_id:
            functions = await clients.list_functions(billing=read_mode)
            assert receipt.function_id not in {item["id"] for item in functions}
        if receipt.flow_id:
            flows = await clients.list_flows(billing=read_mode)
            assert receipt.flow_id not in {item["id"] for item in flows}
        if receipt.trigger_id:
            with pytest.raises(CloudResourceNotFound):
                await clients.get_trigger(receipt.trigger_id, billing=read_mode)
        if receipt.action_ids:
            actions = await clients.list_action_definitions(billing=read_mode)
            assert set(receipt.action_ids).isdisjoint(item["id"] for item in actions)

    try:
        assert (
            current_scope().origin == scope.origin
        ), "Sandbox smoke origin changed before deployment"
        deployed = deploy(source, device_free=True, billing=Billing.PAID, yes=True)
        print(
            f"deployment: function={deployed.function_id} version={deployed.version} "
            f"flow={deployed.flow_id} trigger={deployed.trigger_id} "
            f"actions={deployed.action_ids}",
            flush=True,
        )
        assert deployed.status == "created"
        assert deployed.endpoint and deployed.trigger_id == deployed.endpoint.trigger_id
        assert deployed.flow_id and deployed.action_ids
        if webhook:
            assert len(deployed.action_ids) == 2
        version = asyncio.run(
            clients.get_version(
                deployed.function_id, deployed.version, billing=read_mode
            )
        )
        assert version["status"] == "ready"

        try:
            run = script["ping"].remote.fire(message="smoke", billing=Billing.PAID)
            print(f"invocation={run.invocation_id}", flush=True)
            assert run.invocation_id
            result = run.result()
        except SandboxRunError as exc:
            print(
                f"run failed: invocation={exc.invocation_id} execution={exc.execution_id}",
                flush=True,
            )
            raise
        print(f"execution={run.id} duration_s={run.duration}", flush=True)
        assert run.id and run.snapshot["status"] == "success"
        assert type(result) is script["Pong"]
        assert result.reply == "pong: smoke"
        if webhook:
            event_id = run.snapshot["result"]["steps"][-1]["body"]["eventId"]

            async def verify_delivery():
                async with asyncio.timeout(60), cloud_client(billing=read_mode) as sdk:
                    while True:
                        feed = await sdk.webhooks.deliveries.list(event_id=event_id)
                        if any(
                            item.endpoint_id == webhook and item.status == "success"
                            for item in feed.items
                        ):
                            return
                        await asyncio.sleep(1)

            asyncio.run(verify_delivery())
        # An identical deploy must preserve the running binding, even when paused.
        script["ping"].remote.disable(billing=read_mode)
        repeated = deploy(source, device_free=True, billing=Billing.PAID, yes=True)
        assert (
            repeated.status == "unchanged"
            and repeated.action_ids == deployed.action_ids
        )
    finally:
        print("usage/cost: not exposed by the run handle; check billing", flush=True)
        # Read the on-disk journal even when deploy raised before returning. Never
        # infer ownership from a remote name or replay an uncertain paid creation.
        receipt = store.read()
        if receipt is not None:
            print(
                f"cleanup receipt: function={receipt.function_id} flow={receipt.flow_id} "
                f"trigger={receipt.trigger_id} actions={receipt.action_ids}",
                flush=True,
            )
            try:
                asyncio.run(
                    cleanup_owned(receipt, delete_function=True, billing=read_mode)
                )
                asyncio.run(verify_removed(receipt))
            except BaseException:
                print(
                    "cleanup: incomplete or unverified; retain the receipt", flush=True
                )
                raise
            else:
                print("cleanup: absence verified for all recorded IDs", flush=True)
        else:
            print("cleanup: no ownership receipt; no deletion attempted", flush=True)
