"""Offline SDK transport contracts for sandbox workflows and compute functions."""

import asyncio
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import BaseModel, Field

from mobilerun.errors.cloud import BillingError, CloudAPIError
from mobilerun.models.billing import Billing
from mobilerun.sandbox import clients

DOCUMENT = {
    "version": 1,
    "name": "flow",
    "on": {"custom": {}},
    "steps": [
        {
            "id": "function",
            "name": "action",
            "uses": "functions.run",
            "with": {"functionId": "fn"},
        }
    ],
}


def flow_sdk():
    return SimpleNamespace(
        workflows=SimpleNamespace(
            action_catalog=SimpleNamespace(
                list=AsyncMock(
                    side_effect=[
                        {"items": [], "pagination": {"hasNext": True}},
                        {
                            "items": [{"stepName": "functions.run", "id": "catalog"}],
                            "pagination": {"hasNext": False},
                        },
                    ]
                )
            ),
            flows=SimpleNamespace(
                update=AsyncMock(return_value={"data": {"enabled": False}})
            ),
        )
    )


@pytest.mark.anyio
async def test_catalog_uses_stable_step_name_and_all_pages():
    client = flow_sdk()
    assert await clients._catalog_entry_id(client, "functions.run") == "catalog"
    assert client.workflows.action_catalog.list.await_count == 2


@pytest.mark.anyio
@pytest.mark.parametrize(
    "step,count,reason",
    [("functions.run", 0, "no entries"), ("webhooks.send", 2, "more than one entry")],
)
async def test_catalog_entry_errors_are_specific_and_value_safe(step, count, reason):
    from mobilerun.errors.base import is_value_safe

    client = flow_sdk()
    client.workflows.action_catalog.list = AsyncMock(
        return_value={
            "items": [{"stepName": step, "id": "private-catalog-id"}] * count,
            "pagination": {"hasNext": False},
        }
    )
    with pytest.raises(clients.FlowCapabilityError) as caught:
        await clients._catalog_entry_id(client, step)
    assert str(caught.value) == f"Workflow action catalog: found {reason} for {step}"
    assert caught.value.code == "request_failed" and is_value_safe(caught.value)


@pytest.mark.anyio
async def test_disable_is_free_and_uses_update():
    client = flow_sdk()
    assert await clients.disable_flow(
        "flow", billing=Billing.NO_COMMERCE, client=client
    ) == {"enabled": False}
    client.workflows.flows.update.assert_awaited_once_with("flow", enabled=False)


def test_enable_rejects_before_factory(monkeypatch):
    factory = AsyncMock()
    monkeypatch.setattr(clients, "cloud_client", factory)
    with pytest.raises(BillingError):
        clients.enable_flow("flow", billing=Billing.NO_COMMERCE)
    factory.assert_not_called()


@pytest.mark.anyio
async def test_create_document_stages_and_verifies_disabled():
    client = flow_sdk()
    w = client.workflows
    w.triggers = SimpleNamespace(create=AsyncMock(return_value={"data": {"id": "t"}}))
    w.actions = SimpleNamespace(create=AsyncMock(return_value={"data": {"id": "a"}}))
    w.flows.create = AsyncMock(return_value={"data": {"id": "f"}})
    w.flows.retrieve = AsyncMock(return_value={"data": {"id": "f", "enabled": False}})
    doc = {**DOCUMENT, "settings": {"recording": {"mode": "off"}}}
    result = await clients.create_document(
        doc, "key", billing=Billing.PAID, client=client
    )
    assert result == {"flowId": "f", "triggerId": "t", "actionIds": ["a"]}
    assert w.flows.create.await_args.kwargs["enabled"] is False
    w.flows.retrieve.assert_awaited_once_with("f")


@pytest.mark.anyio
async def test_unsupported_document_fails_before_client_creation(monkeypatch):
    factory = AsyncMock()
    monkeypatch.setattr(clients, "cloud_client", factory)
    with pytest.raises(clients.FlowCapabilityError):
        await clients.create_document({"unknown": True}, "key", billing=Billing.PAID)
    factory.assert_not_called()


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["transport", "missing-id"])
async def test_action_failure_preserves_trigger_for_recovery(failure):
    client = flow_sdk()
    client.workflows.triggers = SimpleNamespace(
        create=AsyncMock(return_value={"data": {"id": "t"}})
    )
    action = AsyncMock(return_value={"data": {}})
    if failure == "transport":
        action.side_effect = httpx.ConnectError("secret")
    client.workflows.actions = SimpleNamespace(create=action)
    with pytest.raises(clients.FlowStagingError) as error:
        await clients.create_document(
            DOCUMENT, "key", billing=Billing.PAID, client=client
        )
    assert error.value.trigger_id == "t"
    assert error.value.action_ids == ()
    assert "secret" not in str(error.value)


@pytest.mark.anyio
async def test_read_wrappers_forward_and_unwrap():
    w = SimpleNamespace(
        flows=SimpleNamespace(
            retrieve_document=AsyncMock(return_value={"data": {"name": "x"}}),
            actions=SimpleNamespace(
                list=AsyncMock(return_value={"data": [{"actionId": "a"}]})
            ),
        ),
        triggers=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"id": "t"}})
        ),
        executions=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"id": "e"}})
        ),
    )
    c = SimpleNamespace(workflows=w)
    assert await clients.get_document("f", billing=Billing.NO_COMMERCE, client=c) == {
        "name": "x"
    }
    assert await clients.list_actions("f", billing=Billing.NO_COMMERCE, client=c) == [
        {"actionId": "a"}
    ]
    assert await clients.get_trigger("t", billing=Billing.NO_COMMERCE, client=c) == {
        "id": "t"
    }
    assert await clients.get_execution("e", billing=Billing.NO_COMMERCE, client=c) == {
        "id": "e"
    }


@pytest.mark.anyio
async def test_paginated_reads_and_execution_order():
    pages = [
        {"items": [{"id": "new"}], "pagination": {"hasNext": True}},
        {"items": [{"id": "old"}], "pagination": {"hasNext": False}},
    ]
    w = SimpleNamespace(
        flows=SimpleNamespace(list=AsyncMock(side_effect=pages)),
        executions=SimpleNamespace(list=AsyncMock(side_effect=pages)),
    )
    c = SimpleNamespace(workflows=w)
    assert await clients.list_flows(billing=Billing.NO_COMMERCE, client=c) == [
        {"id": "new"},
        {"id": "old"},
    ]
    assert await clients.list_executions(
        "f", invocation_id="i", limit=1, billing=Billing.NO_COMMERCE, client=c
    ) == [{"id": "new"}]
    assert w.executions.list.await_args.kwargs["order_by_direction"] == "desc"
    assert w.executions.list.await_args.kwargs["invocation_id"] == "i"


@pytest.mark.anyio
async def test_mutations_forward_typed_parameters_and_verify_replacement():
    w = SimpleNamespace(
        flows=SimpleNamespace(
            validate_document=AsyncMock(return_value={"data": {"valid": True}}),
            retrieve=AsyncMock(return_value={"data": {"enabled": False}}),
            replace_document=AsyncMock(return_value={"data": {"id": "f"}}),
            actions=SimpleNamespace(replace=AsyncMock(return_value={"data": []})),
        ),
        triggers=SimpleNamespace(
            create=AsyncMock(return_value={"data": {"id": "t"}}),
            update=AsyncMock(return_value={"data": {"id": "t"}}),
            fire=AsyncMock(return_value={"data": {"id": "i"}}),
        ),
        actions=SimpleNamespace(create=AsyncMock(return_value={"data": {"id": "a"}})),
    )
    c = SimpleNamespace(workflows=w)
    kw = {"billing": Billing.PAID, "client": c}
    assert await clients.validate_document({"name": "x"}, **kw) == {"valid": True}
    assert await clients.replace_document("f", {"name": "x"}, "k", **kw) == {"id": "f"}
    assert w.flows.retrieve.await_count == 2
    assert await clients.create_trigger(
        {"activation": "custom", "name": "t"}, "k", **kw
    ) == {"id": "t"}
    assert await clients.update_trigger("t", {"name": "updated"}, "k", **kw) == {
        "id": "t"
    }
    assert await clients.create_action(
        {"catalog_entry_id": "c", "name": "a"}, "k", **kw
    ) == {"id": "a"}
    assert (
        await clients.replace_flow_actions(
            "f", [{"action_id": "a", "position": 0}], "k", **kw
        )
        == []
    )
    assert await clients.fire_trigger("t", {"n": 1}, "k", **kw) == {"id": "i"}
    w.triggers.fire.assert_awaited_once_with("t", payload={"n": 1}, idempotency_key="k")


@pytest.mark.anyio
async def test_replace_rejects_enabled_flow():
    w = SimpleNamespace(
        flows=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"enabled": True}}),
            replace_document=AsyncMock(),
        )
    )
    with pytest.raises(clients.FlowCapabilityError):
        await clients.replace_document(
            "f", {}, "k", billing=Billing.PAID, client=SimpleNamespace(workflows=w)
        )
    w.flows.replace_document.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["flow", "action", "trigger"])
async def test_deletes_are_free_and_ignore_response(resource):
    method = AsyncMock(return_value={"data": {"deleted": True}})
    w = SimpleNamespace(**{resource + "s": SimpleNamespace(delete=method)})
    result = await getattr(clients, "delete_" + resource)(
        "id", billing=Billing.NO_COMMERCE, client=SimpleNamespace(workflows=w)
    )
    assert result is None
    method.assert_awaited_once_with("id")


FLOW_OPERATIONS = [
    ("list_flows", (), False),
    ("get_document", ("f",), False),
    ("list_actions", ("f",), False),
    ("get_trigger", ("t",), False),
    ("list_executions", ("f",), False),
    ("get_execution", ("e",), False),
    ("disable_flow", ("f",), False),
    ("delete_flow", ("f",), False),
    ("delete_action", ("a",), False),
    ("delete_trigger", ("t",), False),
    ("validate_document", ({},), False),
    ("create_document", (DOCUMENT, "k"), True),
    ("replace_document", ("f", {}, "k"), True),
    ("enable_flow", ("f",), True),
    ("create_trigger", ({"activation": "custom", "name": "t"}, "k"), True),
    ("update_trigger", ("t", {"name": "t"}, "k"), True),
    ("create_action", ({"catalog_entry_id": "c", "name": "a"}, "k"), True),
    ("replace_flow_actions", ("f", [], "k"), True),
    ("fire_trigger", ("t", {}, "k"), True),
]


def managed_flow_sdk():
    client = SimpleNamespace()
    client.closed = False

    class Managed:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *args):
            client.closed = True

    client.manager = Managed()
    page = {"items": [], "pagination": {"hasNext": False}}
    record = {"data": {"id": "id", "enabled": False}}

    def method():
        return AsyncMock(return_value=record)

    client.workflows = SimpleNamespace(
        action_catalog=SimpleNamespace(
            list=AsyncMock(
                return_value={
                    "items": [{"stepName": "functions.run", "id": "catalog"}],
                    "pagination": {"hasNext": False},
                }
            )
        ),
        flows=SimpleNamespace(
            list=AsyncMock(return_value=page),
            retrieve=method(),
            retrieve_document=method(),
            validate_document=method(),
            replace_document=method(),
            create=method(),
            update=method(),
            delete=method(),
            actions=SimpleNamespace(
                list=AsyncMock(return_value={"data": []}),
                replace=AsyncMock(return_value={"data": []}),
            ),
        ),
        actions=SimpleNamespace(create=method(), delete=method()),
        triggers=SimpleNamespace(
            retrieve=method(),
            create=method(),
            update=method(),
            fire=method(),
            delete=method(),
        ),
        executions=SimpleNamespace(
            list=AsyncMock(return_value=page), retrieve=method()
        ),
    )
    return client


@pytest.mark.anyio
@pytest.mark.parametrize("name,args,paid", FLOW_OPERATIONS)
async def test_billing_matrix(name, args, paid, monkeypatch):
    c = managed_flow_sdk()

    factory = Mock(return_value=c.manager)
    monkeypatch.setattr(clients, "cloud_client", factory)
    fn = getattr(clients, name)
    with pytest.raises(TypeError):
        fn(*args)
    with pytest.raises(BillingError):
        fn(*args, billing="invalid")
    factory.assert_not_called()
    for mode in (Billing.NO_COMMERCE, Billing.INCLUDED, Billing.PAID):
        if paid and mode is not Billing.PAID:
            with pytest.raises(BillingError):
                fn(*args, billing=mode)
            factory.assert_not_called()
        else:
            await fn(*args, billing=mode, client=c)


@pytest.mark.anyio
@pytest.mark.parametrize("name,args,paid", FLOW_OPERATIONS)
async def test_client_ownership(name, args, paid, monkeypatch):
    injected = managed_flow_sdk()
    await getattr(clients, name)(*args, billing=Billing.PAID, client=injected)
    assert not injected.closed
    owned = managed_flow_sdk()
    monkeypatch.setattr(clients, "cloud_client", lambda **kwargs: owned.manager)
    await getattr(clients, name)(*args, billing=Billing.PAID)
    assert owned.closed


@pytest.mark.anyio
async def test_cloud_error_is_sanitized_and_owned_client_closed(monkeypatch):
    from mobilerun.errors.cloud import CloudConnectionError

    c = managed_flow_sdk()
    c.workflows.flows.list.side_effect = httpx.ConnectError("secret-url")
    monkeypatch.setattr(clients, "cloud_client", lambda **kwargs: c.manager)
    with pytest.raises(CloudConnectionError) as error:
        await clients.list_flows(billing=Billing.NO_COMMERCE)
    assert "secret-url" not in str(error.value)
    assert c.closed


@pytest.mark.anyio
async def test_execution_list_fetches_all_pages_then_limits_and_uses_descending_order():
    pages = [
        {"items": [{"id": "1"}, {"id": "2"}], "pagination": {"hasNext": True}},
        {"items": [{"id": "3"}], "pagination": {"hasNext": False}},
    ]
    method = AsyncMock(side_effect=pages)
    client = SimpleNamespace(
        workflows=SimpleNamespace(executions=SimpleNamespace(list=method))
    )
    assert await clients.list_executions(
        "f", limit=3, billing=Billing.NO_COMMERCE, client=client
    ) == [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    assert method.await_args_list[1].kwargs["page"] == 2
    assert method.await_args_list[0].kwargs["order_by"] == "startedAt"
    assert method.await_args_list[0].kwargs["order_by_direction"] == "desc"


FAILURE_PATHS = {
    "list_flows": "flows.list",
    "get_document": "flows.retrieve_document",
    "list_actions": "flows.actions.list",
    "get_trigger": "triggers.retrieve",
    "list_executions": "executions.list",
    "get_execution": "executions.retrieve",
    "disable_flow": "flows.update",
    "delete_flow": "flows.delete",
    "delete_action": "actions.delete",
    "delete_trigger": "triggers.delete",
    "validate_document": "flows.validate_document",
    "create_document": "action_catalog.list",
    "replace_document": "flows.retrieve",
    "enable_flow": "flows.update",
    "create_trigger": "triggers.create",
    "update_trigger": "triggers.update",
    "create_action": "actions.create",
    "replace_flow_actions": "flows.actions.replace",
    "fire_trigger": "triggers.fire",
}


@pytest.mark.anyio
@pytest.mark.parametrize("name,args,paid", FLOW_OPERATIONS)
async def test_client_ownership_on_transport_failure(name, args, paid, monkeypatch):
    for injected in (False, True):
        c = managed_flow_sdk()
        method = c.workflows
        for part in FAILURE_PATHS[name].split("."):
            method = getattr(method, part)
        method.side_effect = httpx.ConnectError("secret-url")

        monkeypatch.setattr(clients, "cloud_client", Mock(return_value=c.manager))
        with pytest.raises(CloudAPIError) as error:
            await getattr(clients, name)(
                *args, billing=Billing.PAID, client=c if injected else None
            )
        assert "secret-url" not in str(error.value)
        assert c.closed is not injected


@pytest.mark.anyio
@pytest.mark.parametrize("bad", [None, [], "bad", 1])
async def test_malformed_verification_preserves_all_created_ids(bad):
    c = managed_flow_sdk()
    c.workflows.flows.retrieve.return_value = {"data": bad}
    with pytest.raises(clients.FlowStagingError) as error:
        await clients.create_document(DOCUMENT, "key", billing=Billing.PAID, client=c)
    assert error.value.trigger_id == "id"
    assert error.value.action_ids == ("id",)
    assert error.value.flow_id == "id"


@pytest.mark.anyio
@pytest.mark.parametrize("failure_stage", [0, 1, 2, 3])
async def test_staging_failure_preserves_each_created_id(failure_stage):
    c = managed_flow_sdk()
    c.workflows.triggers.create.return_value = {"data": {"id": "t"}}
    action_results = [
        {"data": {"id": "a0"}},
        {"data": {"id": "a1"}},
    ]
    c.workflows.flows.create.return_value = {"data": {"id": "f"}}
    if failure_stage < 2:
        action_results[failure_stage] = httpx.ConnectError("private")
    elif failure_stage == 2:
        c.workflows.flows.create.side_effect = httpx.ConnectError("private")
    else:
        c.workflows.flows.retrieve.side_effect = httpx.ConnectError("private")
    c.workflows.actions.create.side_effect = action_results
    doc = {**DOCUMENT, "steps": DOCUMENT["steps"] * 2}
    with pytest.raises(clients.FlowStagingError) as error:
        await clients.create_document(doc, "key", billing=Billing.PAID, client=c)
    assert error.value.trigger_id == "t"
    assert error.value.action_ids == tuple(["a0", "a1"][:failure_stage])
    assert error.value.flow_id == ("f" if failure_stage == 3 else None)


def test_flow_errors_are_safe_rendering():
    from mobilerun.errors.base import is_value_safe
    from mobilerun.sandbox.errors import FlowCapabilityError, FlowStagingError

    assert is_value_safe(
        FlowCapabilityError(operation="Stage flow", code="request_failed")
    )
    assert is_value_safe(FlowStagingError("trigger", ["action"], "flow"))


@pytest.mark.anyio
async def test_document_validation_is_no_commerce():
    c = managed_flow_sdk()
    await clients.validate_document({}, billing=Billing.NO_COMMERCE, client=c)
    c.workflows.flows.validate_document.assert_awaited_once_with(document={})


@pytest.mark.anyio
async def test_update_flow_devices_forwards_empty_list_and_key():
    client = flow_sdk()
    await clients.update_flow_devices(
        "flow", [], "key", billing=Billing.PAID, client=client
    )
    client.workflows.flows.update.assert_awaited_once_with(
        "flow", device_ids=[], idempotency_key="key"
    )


def test_update_flow_devices_requires_paid_before_client():
    with pytest.raises(BillingError):
        clients.update_flow_devices("flow", [], "key", billing=Billing.NO_COMMERCE)


def function_sdk():
    versions = SimpleNamespace(
        create=AsyncMock(), retrieve=AsyncMock(), list=AsyncMock()
    )
    funcs = SimpleNamespace(
        create=AsyncMock(),
        source=AsyncMock(),
        delete=AsyncMock(),
        list=AsyncMock(),
        versions=versions,
    )
    logs = AsyncMock()
    sdk = SimpleNamespace(
        compute=SimpleNamespace(functions=funcs, runs=SimpleNamespace(logs=logs))
    )
    return sdk


class Owned:
    def __init__(self, sdk):
        self.sdk = sdk
        self.closed = False

    async def __aenter__(self):
        return self.sdk

    async def __aexit__(self, *args):
        self.closed = True


class Item(BaseModel):
    latest_version: int = Field(alias="latestVersion")


class Envelope(BaseModel):
    data: dict


def test_list_functions_paginates_models_and_empty_page():
    sdk = function_sdk()
    sdk.compute.functions.list.side_effect = [
        Envelope(
            data={"items": [Item(latestVersion=3)], "pagination": {"hasNext": True}}
        ),
        {"data": {"items": [], "pagination": {"hasNext": True}}},
        {"data": {"items": [{"id": "two"}], "pagination": {"hasNext": False}}},
    ]
    assert asyncio.run(
        clients.list_functions(billing=Billing.NO_COMMERCE, client=sdk)
    ) == [
        {"latestVersion": 3},
        {"id": "two"},
    ]
    assert sdk.compute.functions.list.call_count == 3


def test_list_versions_paginates_and_omits_null_items():
    sdk = function_sdk()
    sdk.compute.functions.versions.list.side_effect = [
        {"data": {"items": [None, {"version": 1}], "pagination": {"hasNext": True}}},
        {"data": {"items": [{"version": 2}], "pagination": {"hasNext": False}}},
    ]
    assert asyncio.run(
        clients.list_versions("f", billing=Billing.NO_COMMERCE, client=sdk)
    ) == [{"version": 1}, {"version": 2}]
    sdk.compute.functions.versions.list.assert_any_await("f", page=2, page_size=100)


@pytest.mark.parametrize(
    "name,args,path,expected_call",
    [
        ("get_version", ("f", 2), "versions.retrieve", ((2,), {"function_id": "f"})),
        (
            "create_function",
            ("n", "src", "key"),
            "create",
            ((), {"name": "n", "source": "src", "idempotency_key": "key"}),
        ),
        (
            "create_version",
            ("f", "src", "key"),
            "versions.create",
            (("f",), {"source": "src", "idempotency_key": "key"}),
        ),
        ("delete_function", ("f",), "delete", (("f",), {})),
    ],
)
def test_sdk_dispatch(name, args, path, expected_call):
    sdk = function_sdk()
    method = sdk_method(sdk, path)
    method.return_value = {"data": {"id": "x"}}
    result = asyncio.run(
        getattr(clients, name)(*args, billing=Billing.PAID, client=sdk)
    )
    assert result == (None if name == "delete_function" else {"id": "x"})
    method.assert_awaited_once_with(*expected_call[0], **expected_call[1])


def test_get_source():
    sdk = function_sdk()
    sdk.compute.functions.source.return_value = {"data": {"source": "hello"}}
    assert (
        asyncio.run(clients.get_source("f", 7, billing=Billing.NO_COMMERCE, client=sdk))
        == "hello"
    )
    sdk.compute.functions.source.assert_awaited_once_with("f", version=7)


@pytest.mark.parametrize(
    "name,args",
    [
        ("list_functions", ()),
        ("list_versions", ("f",)),
        ("get_version", ("f", 1)),
        ("get_source", ("f", 1)),
        ("create_function", ("n", "s", "k")),
        ("create_version", ("f", "s", "k")),
        ("delete_function", ("f",)),
        ("run_logs", ("r",)),
    ],
)
@pytest.mark.parametrize("billing", [None, "invalid"])
def test_invalid_billing_rejected_before_factory(monkeypatch, name, args, billing):
    factory = Mock()
    monkeypatch.setattr(clients, "cloud_client", factory)
    with pytest.raises(BillingError):
        asyncio.run(getattr(clients, name)(*args, billing=billing))
    factory.assert_not_called()


@pytest.mark.parametrize(
    "name,args",
    [("create_function", ("n", "s", "k")), ("create_version", ("f", "s", "k"))],
)
@pytest.mark.parametrize("billing", [Billing.NO_COMMERCE, Billing.INCLUDED])
def test_paid_creations_reject_lower_modes_before_factory(
    monkeypatch, name, args, billing
):
    factory = Mock()
    monkeypatch.setattr(clients, "cloud_client", factory)
    with pytest.raises(BillingError):
        asyncio.run(getattr(clients, name)(*args, billing=billing))
    factory.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_owned_client_closed_and_transport_errors_sanitized(monkeypatch, fails):
    sdk = function_sdk()
    sdk.compute.functions.delete.return_value = None
    if fails:
        sdk.compute.functions.delete.side_effect = httpx.ConnectError(
            "secret-url", request=httpx.Request("GET", "https://secret.test/token")
        )
    owned = Owned(sdk)
    monkeypatch.setattr(clients, "cloud_client", Mock(return_value=owned))
    if fails:
        with pytest.raises(Exception) as caught:
            asyncio.run(clients.delete_function("f", billing=Billing.NO_COMMERCE))
        assert "secret" not in str(caught.value)
        assert caught.value.__suppress_context__
    else:
        asyncio.run(clients.delete_function("f", billing=Billing.NO_COMMERCE))
    assert owned.closed


def test_injected_client_not_closed():
    sdk = function_sdk()
    sdk.close = AsyncMock()
    asyncio.run(clients.delete_function("f", billing=Billing.NO_COMMERCE, client=sdk))
    sdk.close.assert_not_awaited()


def test_run_logs_uses_separate_unauthenticated_client(monkeypatch):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?secret=token"}
    }
    seen = []
    transport = httpx.MockTransport(
        lambda request: (seen.append(request), httpx.Response(200, text="logs"))[1]
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        clients.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    assert (
        asyncio.run(clients.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
        == "logs"
    )
    assert len(seen) == 1 and "authorization" not in seen[0].headers


def test_run_logs_http_error_does_not_expose_signed_url(monkeypatch):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?secret=token"}
    }
    transport = httpx.MockTransport(
        lambda request: httpx.Response(403, request=request)
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        clients.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    with pytest.raises(Exception) as caught:
        asyncio.run(clients.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret" not in str(caught.value) and "signed.example" not in str(
        caught.value
    )
    assert caught.value.__suppress_context__


@pytest.mark.parametrize(
    "url",
    [
        "https://host:secret-token/file",
        {"secret-token": "malformed"},
        "ftp://host/secret-token",
    ],
)
def test_run_logs_invalid_url_never_exposes_secret(url):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {"data": {"url": url}}
    with pytest.raises(Exception) as caught:
        asyncio.run(clients.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret-token" not in str(caught.value)
    assert "secret-token" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize("error_type", [httpx.CookieConflict, httpx.StreamError])
def test_run_logs_other_httpx_errors_are_sanitized(monkeypatch, error_type):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {"data": {"url": "https://example.test"}}
    transport = httpx.MockTransport(
        lambda request: (_ for _ in ()).throw(error_type("secret-token"))
    )
    real = httpx.AsyncClient
    monkeypatch.setattr(
        clients.httpx,
        "AsyncClient",
        lambda **kwargs: real(transport=transport, **kwargs),
    )
    with pytest.raises(Exception) as caught:
        asyncio.run(clients.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
    assert "secret-token" not in str(caught.value)
    assert "secret-token" not in "".join(traceback.format_exception(caught.value))


READ_OPERATIONS = [
    ("list_functions", (), "list"),
    ("list_versions", ("f",), "versions.list"),
    ("get_version", ("f", 1), "versions.retrieve"),
    ("get_source", ("f", 1), "source"),
    ("delete_function", ("f",), "delete"),
]
ALL_OPERATIONS = READ_OPERATIONS + [
    ("create_function", ("n", "s", "k"), "create"),
    ("create_version", ("f", "s", "k"), "versions.create"),
    ("run_logs", ("r",), "runs.logs"),
]


def sdk_method(sdk, path):
    target = sdk.compute if path.startswith("runs.") else sdk.compute.functions
    for part in path.split("."):
        target = getattr(target, part)
    return target


@pytest.mark.parametrize("name,args,path", ALL_OPERATIONS)
def test_billing_required_before_factory(monkeypatch, name, args, path):
    factory = Mock()
    monkeypatch.setattr(clients, "cloud_client", factory)
    with pytest.raises(TypeError):
        getattr(clients, name)(*args)
    factory.assert_not_called()


@pytest.mark.parametrize("name,args,path", READ_OPERATIONS)
@pytest.mark.parametrize("billing", list(Billing))
def test_read_and_delete_accept_all_billing(name, args, path, billing):
    sdk = function_sdk()
    method = sdk_method(sdk, path)
    method.return_value = {
        "data": {"items": [], "pagination": {"hasNext": False}, "source": "source"}
    }
    asyncio.run(getattr(clients, name)(*args, billing=billing, client=sdk))
    method.assert_awaited_once()


@pytest.mark.parametrize("name,args,path", ALL_OPERATIONS)
def test_sdk_errors_are_sanitized_for_every_operation(name, args, path):
    sdk = function_sdk()
    sdk.close = AsyncMock()
    sdk_method(sdk, path).side_effect = httpx.ConnectError(
        "secret-url", request=httpx.Request("GET", "https://secret.test/token")
    )
    with pytest.raises(Exception) as caught:
        asyncio.run(getattr(clients, name)(*args, billing=Billing.PAID, client=sdk))
    assert "secret" not in str(caught.value)
    assert caught.value.__suppress_context__
    sdk.close.assert_not_awaited()


def test_sdk_transport_uses_generated_endpoints_and_idempotency():
    from mobilerun_sdk import AsyncMobilerun

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "data": {
                    "id": "id",
                    "source": "hello",
                    "items": [],
                    "pagination": {"hasNext": False},
                }
            },
        )

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            async with AsyncMobilerun(
                api_key="fake", base_url="https://example.test/v1", http_client=http
            ) as sdk:
                await clients.list_functions(billing=Billing.NO_COMMERCE, client=sdk)
                await clients.list_versions(
                    "f", billing=Billing.NO_COMMERCE, client=sdk
                )
                await clients.get_version(
                    "f", 2, billing=Billing.NO_COMMERCE, client=sdk
                )
                await clients.get_source(
                    "f", 2, billing=Billing.NO_COMMERCE, client=sdk
                )
                await clients.create_function(
                    "n", "src", "key1", billing=Billing.PAID, client=sdk
                )
                await clients.create_version(
                    "f", "src", "key2", billing=Billing.PAID, client=sdk
                )
                await clients.delete_function(
                    "f", billing=Billing.NO_COMMERCE, client=sdk
                )

    asyncio.run(exercise())
    assert [r.url.path for r in requests] == [
        "/v1/compute/functions",
        "/v1/compute/functions/f/versions",
        "/v1/compute/functions/f/versions/2",
        "/v1/compute/functions/f/source",
        "/v1/compute/functions",
        "/v1/compute/functions/f/versions",
        "/v1/compute/functions/f",
    ]
    assert requests[4].headers["idempotency-key"] == "key1"
    assert requests[5].headers["idempotency-key"] == "key2"


@pytest.mark.parametrize("billing", list(Billing))
@pytest.mark.parametrize("fail_download", [False, True])
def test_logs_owns_both_clients_and_accepts_all_billing(
    monkeypatch, billing, fail_download
):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {
        "data": {"url": "https://signed.example/log?token=secret"}
    }
    owned = Owned(sdk)
    monkeypatch.setattr(clients, "cloud_client", Mock(return_value=owned))
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(500 if fail_download else 200, text="ok")

    transport = httpx.MockTransport(respond)
    actual = httpx.AsyncClient
    http_clients = []

    def make_client(**kwargs):
        downloader = actual(transport=transport, **kwargs)
        http_clients.append(downloader)
        return downloader

    monkeypatch.setattr(clients.httpx, "AsyncClient", make_client)
    if fail_download:

        with pytest.raises(Exception) as caught:
            asyncio.run(clients.run_logs("r", billing=billing))
        formatted = "".join(traceback.format_exception(caught.value))
        assert "token=secret" not in formatted and "signed.example" not in formatted
    else:
        assert asyncio.run(clients.run_logs("r", billing=billing)) == "ok"
    assert owned.closed and http_clients[0].is_closed
    assert "authorization" not in seen[0].headers


def test_logs_follow_redirect_without_credentials(monkeypatch):
    sdk = function_sdk()
    sdk.compute.runs.logs.return_value = {"data": {"url": "https://signed.example/a"}}
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(302, headers={"location": "https://other.example/b"})
            if len(requests) == 1
            else httpx.Response(200, text="ok")
        )

    transport = httpx.MockTransport(respond)
    actual = httpx.AsyncClient
    monkeypatch.setattr(
        clients.httpx,
        "AsyncClient",
        lambda **kwargs: actual(transport=transport, **kwargs),
    )
    assert (
        asyncio.run(clients.run_logs("r", billing=Billing.NO_COMMERCE, client=sdk))
        == "ok"
    )
    assert len(requests) == 2
    assert all("authorization" not in request.headers for request in requests)
