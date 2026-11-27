"""Opt-in, potentially paid, device-free sandbox endpoint lifecycle."""

import asyncio
import json
import os
from base64 import b64decode
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
from textwrap import dedent
from urllib.parse import urlsplit
from uuid import UUID, uuid4
from warnings import warn

import httpx
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
    from mobilerun.sandbox.receipts import current_scope

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
            "Set MOBILERUN_SMOKE_WEBHOOK_URL to a dedicated https://webhook.site/<uuid> receiver you control"
        )
    parsed = urlsplit(webhook)
    try:
        token = str(UUID(parsed.path.strip("/")))
    except ValueError:
        pytest.fail("Expected a webhook.site receiver UUID", pytrace=False)
    if parsed.scheme != "https" or parsed.netloc != "webhook.site":
        pytest.fail("Expected a dedicated HTTPS webhook.site receiver", pytrace=False)
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
        _exercise_endpoint(
            tmp_path, monkeypatch, webhook=endpoint.id, receiver_token=token
        )
    finally:
        asyncio.run(hooks.delete(endpoint.id))
        with pytest.raises(CloudResourceNotFound):
            asyncio.run(hooks.get(endpoint.id))


async def _cleanup_saved_copy(
    folder, execution_id, *, wait_for_copy, get_execution, clean, invocation_id=None
):
    try:
        await wait_for_copy()
    except TimeoutError:
        # The execution may already have succeeded while its separate copy job
        # is still queued. Only the matching delivery can confirm completion.
        deliveries = []
        if execution_id is not None:
            try:
                async with asyncio.timeout(10):
                    execution = await get_execution(execution_id)
                deliveries = execution.get("deliveries") or []
            except Exception:
                pass  # An unavailable status cannot confirm cleanup.
        if not isinstance(deliveries, list) or not any(
            isinstance(delivery, dict)
            and delivery.get("destination") == "mobilerun"
            and delivery.get("artifact") == "file"
            and delivery.get("folder") == folder
            and delivery.get("filename") == "pixel.png"
            and delivery.get("status") in ("succeeded", "failed", "cancelled")
            for delivery in deliveries
        ):
            warn(
                f"Saved-file cleanup unconfirmed; folder={folder}; "
                f"execution={execution_id or 'unknown'}; "
                f"invocation={invocation_id or 'unknown'}. Folder left for manual cleanup.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
    await clean()


def test_device_free_sandbox_files(real_env, tmp_path, monkeypatch):
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Set MOBILERUN_API_KEY for the sandbox smoke")
    pytest.importorskip("mobilerun_functions")
    pytest.importorskip("mobilerun_sdk.resources.routine_files")

    from mobilerun.clients.cloud.client import cloud_client
    from mobilerun.models.billing import Billing
    from mobilerun.sandbox import clients, cloud, deploy
    from mobilerun.sandbox.cleanup import cleanup_owned
    from mobilerun.sandbox.receipts import ReceiptStore

    scope = _verified_smoke_scope()
    name = f"sandbox-files-smoke-{uuid4().hex}"
    folder = f"/{name}"
    image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNgAAAAAgABSK+kcQAAAABJRU5ErkJggg=="
    source = tmp_path / "image.py"
    monkeypatch.setenv("MOBILERUN_SANDBOX_DIRECTORY", str(tmp_path / "sandbox"))
    store = ReceiptStore(scope, name)
    read_mode = Billing.NO_COMMERCE
    file_options = f", files_folder={folder!r}, files={{'pixel.png': 'image'}}"
    template = dedent(f"""\
        from pydantic import BaseModel
        from mobilerun.sandbox import cloud

        class Input(BaseModel):
            pass

        class Output(BaseModel):
            image: str

        @cloud({name!r}FILE_OPTIONS)
        def task(value: Input) -> Output:
            return Output(image={image!r})
    """)

    async def wait_for_saved_file(**filters):
        async with asyncio.timeout(60), cloud_client(billing=read_mode) as sdk:
            while True:
                page = await sdk.routine_files.list(folder=folder, **filters)
                if page.items:
                    (saved,) = page.items
                    return saved
                await asyncio.sleep(1)

    async def clean_saved_files():
        async with cloud_client(billing=read_mode) as sdk:
            await sdk.routine_files.delete_many(folder=folder)
            assert not (await sdk.routine_files.list(folder=folder)).items

    async def get_execution(execution_id):
        return await clients.get_execution(execution_id, billing=read_mode)

    print(
        f"sandbox files smoke: folder={folder} receipt={store.receipt_path}", flush=True
    )
    fired = False
    saw_copy = False
    run = None
    try:
        source.write_text(
            template.replace("FILE_OPTIONS", file_options), encoding="utf-8"
        )
        deployed = deploy(source, device_free=True, billing=Billing.PAID, yes=True)
        assert deployed.flow_id and len(deployed.action_ids) == 2
        fired = True
        run = cloud[name].fire({}, billing=Billing.PAID)
        assert run.result() == {"image": image}
        assert run.snapshot["status"] == "success"
        saved = asyncio.run(wait_for_saved_file(execution_id=run.id))
        saw_copy = True
        assert (saved.filename, saved.mime_type, saved.folder, saved.size_bytes) == (
            "pixel.png",
            "image/png",
            folder,
            len(b64decode(image)),
        )
        print(f"saved file={saved.id} execution={run.id}", flush=True)

        source.write_text(template.replace("FILE_OPTIONS", ""), encoding="utf-8")
        updated = deploy(source, device_free=True, billing=Billing.PAID, yes=True)
        assert updated.flow_id == deployed.flow_id
        document = asyncio.run(clients.get_document(updated.flow_id, billing=read_mode))
        assert document["settings"].get("delivery") is None
        assert [step["uses"] for step in document["steps"]] == ["functions.run"]
    finally:
        try:
            if fired:
                if saw_copy:
                    asyncio.run(clean_saved_files())
                else:
                    asyncio.run(
                        _cleanup_saved_copy(
                            folder,
                            run.snapshot.get("id") if run is not None else None,
                            wait_for_copy=wait_for_saved_file,
                            get_execution=get_execution,
                            clean=clean_saved_files,
                            invocation_id=(
                                run.invocation_id if run is not None else None
                            ),
                        )
                    )
        finally:
            receipt = store.read()
            if receipt is not None:
                asyncio.run(
                    cleanup_owned(receipt, delete_function=True, billing=read_mode)
                )


async def _verify_webhook_capture(token, event_id):
    async with httpx.AsyncClient(
        base_url="https://webhook.site", timeout=10
    ) as receiver:
        for attempt in range(10):
            try:
                response = await receiver.get(
                    f"/token/{token}/requests",
                    params={"page": 1, "per_page": 25, "_": attempt},
                )
                response.raise_for_status()
            except httpx.HTTPError:
                pytest.fail("Could not read webhook.site captures", pytrace=False)
            for request in response.json()["data"]:
                if request.get("method") != "POST":
                    continue
                try:
                    event = json.loads(request.get("content") or "")
                except ValueError:
                    continue
                if not isinstance(event, dict) or event.get("id") != event_id:
                    continue
                if event.get("type") != "workflow.step.emit":
                    pytest.fail("Unexpected captured webhook event type", pytrace=False)
                if event.get("data") != {"result": {"reply": "pong: smoke"}}:
                    pytest.fail("Unexpected captured webhook event data", pytrace=False)
                return
            await asyncio.sleep(1)
    pytest.fail("The sandbox webhook event was not captured", pytrace=False)


def _exercise_endpoint(tmp_path, monkeypatch, webhook, receiver_token=None):
    with TemporaryDirectory(prefix="mobilerun-smoke-source-") as directory:
        _run_lifecycle(
            tmp_path, monkeypatch, webhook, Path(directory) / "ping.py", receiver_token
        )


def _run_lifecycle(tmp_path, monkeypatch, webhook, source, receiver_token):
    if not os.getenv("MOBILERUN_API_KEY"):
        pytest.skip("Set MOBILERUN_API_KEY for the sandbox smoke")
    pytest.importorskip("mobilerun_functions")
    pytest.importorskip("mobilerun_sdk")

    from mobilerun.clients.cloud.client import cloud_client
    from mobilerun.errors.cloud import CloudResourceNotFound
    from mobilerun.models.billing import Billing
    from mobilerun.sandbox import clients, deploy
    from mobilerun.sandbox.cleanup import cleanup_owned
    from mobilerun.sandbox.errors import SandboxRunError
    from mobilerun.sandbox.receipts import ReceiptStore, current_scope

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
            asyncio.run(_verify_webhook_capture(receiver_token, event_id))
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
