"""Cloud workflow SDK boundary tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.errors.cloud import BillingError, CloudAPIError
from mobilerun.models.billing import Billing
from mobilerun.sandbox import clients as flows


def sdk():
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
    client = sdk()
    assert await flows._catalog(client) == "catalog"
    assert client.workflows.action_catalog.list.await_count == 2


@pytest.mark.anyio
async def test_disable_is_free_and_uses_update():
    client = sdk()
    assert await flows.disable_flow(
        "flow", billing=Billing.NO_COMMERCE, client=client
    ) == {"enabled": False}
    client.workflows.flows.update.assert_awaited_once_with("flow", enabled=False)


def test_enable_rejects_before_factory(monkeypatch):
    factory = AsyncMock()
    monkeypatch.setattr(flows, "cloud_client", factory)
    with pytest.raises(BillingError):
        flows.enable_flow("flow", billing=Billing.NO_COMMERCE)
    factory.assert_not_called()


@pytest.mark.anyio
async def test_create_document_stages_and_verifies_disabled():
    client = sdk()
    w = client.workflows
    w.triggers = SimpleNamespace(create=AsyncMock(return_value={"data": {"id": "t"}}))
    w.actions = SimpleNamespace(create=AsyncMock(return_value={"data": {"id": "a"}}))
    w.flows.create = AsyncMock(return_value={"data": {"id": "f"}})
    w.flows.retrieve = AsyncMock(return_value={"data": {"id": "f", "enabled": False}})
    doc = {
        "version": 1,
        "name": "flow",
        "on": {"custom": {}},
        "settings": {"recording": {"mode": "off"}},
        "steps": [
            {
                "id": "function",
                "name": "action",
                "uses": "functions.run",
                "with": {"functionId": "fn"},
            }
        ],
    }
    result = await flows.create_document(
        doc, "key", billing=Billing.PAID, client=client
    )
    assert result == {"flowId": "f", "triggerId": "t", "actionIds": ["a"]}
    assert w.flows.create.await_args.kwargs["enabled"] is False
    w.flows.retrieve.assert_awaited_once_with("f")


@pytest.mark.anyio
async def test_unsupported_document_fails_before_client_creation(monkeypatch):
    factory = AsyncMock()
    monkeypatch.setattr(flows, "cloud_client", factory)
    with pytest.raises(flows.FlowCapabilityError):
        await flows.create_document({"unknown": True}, "key", billing=Billing.PAID)
    factory.assert_not_called()


@pytest.mark.anyio
async def test_failed_action_exposes_created_trigger_without_server_message():
    client = sdk()
    client.workflows.triggers = SimpleNamespace(
        create=AsyncMock(return_value={"data": {"id": "t"}})
    )
    import httpx

    client.workflows.actions = SimpleNamespace(
        create=AsyncMock(side_effect=httpx.ConnectError("secret"))
    )
    doc = {
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
    with pytest.raises(flows.FlowStagingError) as error:
        await flows.create_document(doc, "key", billing=Billing.PAID, client=client)
    assert error.value.trigger_id == "t"
    assert error.value.action_ids == ()
    assert "secret" not in str(error.value)


@pytest.mark.anyio
async def test_malformed_action_response_preserves_trigger_for_recovery():
    client = sdk()
    client.workflows.triggers = SimpleNamespace(
        create=AsyncMock(return_value={"data": {"id": "t"}})
    )
    client.workflows.actions = SimpleNamespace(
        create=AsyncMock(return_value={"data": {}})
    )
    doc = {
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
    with pytest.raises(flows.FlowStagingError) as error:
        await flows.create_document(doc, "key", billing=Billing.PAID, client=client)
    assert error.value.trigger_id == "t"


@pytest.mark.anyio
async def test_read_wrappers_forward_and_unwrap():
    w = SimpleNamespace(
        flows=SimpleNamespace(
            retrieve_document=AsyncMock(return_value={"data": {"name": "x"}}),
            actions=SimpleNamespace(
                list=AsyncMock(return_value={"data": [{"actionId": "a"}]})
            ),
        ),
        actions=SimpleNamespace(retrieve=AsyncMock(return_value={"data": {"id": "a"}})),
        triggers=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"id": "t"}})
        ),
        executions=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"id": "e"}})
        ),
    )
    c = SimpleNamespace(workflows=w)
    assert await flows.get_document("f", billing=Billing.NO_COMMERCE, client=c) == {
        "name": "x"
    }
    assert await flows.list_actions("f", billing=Billing.NO_COMMERCE, client=c) == [
        {"actionId": "a"}
    ]
    assert await flows.get_action("a", billing=Billing.NO_COMMERCE, client=c) == {
        "id": "a"
    }
    assert await flows.get_trigger("t", billing=Billing.NO_COMMERCE, client=c) == {
        "id": "t"
    }
    assert await flows.get_execution("e", billing=Billing.NO_COMMERCE, client=c) == {
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
    assert await flows.list_flows(billing=Billing.NO_COMMERCE, client=c) == [
        {"id": "new"},
        {"id": "old"},
    ]
    assert await flows.list_executions(
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
    assert await flows.validate_document({"name": "x"}, **kw) == {"valid": True}
    assert await flows.replace_document("f", {"name": "x"}, "k", **kw) == {"id": "f"}
    assert w.flows.retrieve.await_count == 2
    assert await flows.create_trigger(
        {"activation": "custom", "name": "t"}, "k", **kw
    ) == {"id": "t"}
    assert await flows.update_trigger("t", {"name": "updated"}, "k", **kw) == {
        "id": "t"
    }
    assert await flows.create_action(
        {"catalog_entry_id": "c", "name": "a"}, "k", **kw
    ) == {"id": "a"}
    assert (
        await flows.replace_flow_actions(
            "f", [{"action_id": "a", "position": 0}], "k", **kw
        )
        == []
    )
    assert await flows.fire_trigger("t", {"n": 1}, "k", **kw) == {"id": "i"}
    w.triggers.fire.assert_awaited_once_with("t", payload={"n": 1}, idempotency_key="k")


@pytest.mark.anyio
async def test_replace_rejects_enabled_flow():
    w = SimpleNamespace(
        flows=SimpleNamespace(
            retrieve=AsyncMock(return_value={"data": {"enabled": True}}),
            replace_document=AsyncMock(),
        )
    )
    with pytest.raises(flows.FlowCapabilityError):
        await flows.replace_document(
            "f", {}, "k", billing=Billing.PAID, client=SimpleNamespace(workflows=w)
        )
    w.flows.replace_document.assert_not_awaited()


@pytest.mark.anyio
@pytest.mark.parametrize("resource", ["flow", "action", "trigger"])
async def test_deletes_are_free_and_ignore_response(resource):
    method = AsyncMock(return_value={"data": {"deleted": True}})
    w = SimpleNamespace(**{resource + "s": SimpleNamespace(delete=method)})
    result = await getattr(flows, "delete_" + resource)(
        "id", billing=Billing.NO_COMMERCE, client=SimpleNamespace(workflows=w)
    )
    assert result is None
    method.assert_awaited_once_with("id")


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
CASES = [
    ("list_flows", (), False),
    ("get_document", ("f",), False),
    ("list_actions", ("f",), False),
    ("get_action", ("a",), False),
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


def shared_client():
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
        actions=SimpleNamespace(retrieve=method(), create=method(), delete=method()),
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
@pytest.mark.parametrize("name,args,paid", CASES)
async def test_billing_matrix(name, args, paid, monkeypatch):
    c = shared_client()
    from unittest.mock import Mock

    factory = Mock(return_value=c.manager)
    monkeypatch.setattr(flows, "cloud_client", factory)
    fn = getattr(flows, name)
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
@pytest.mark.parametrize("name,args,paid", CASES)
async def test_client_ownership(name, args, paid, monkeypatch):
    injected = shared_client()
    await getattr(flows, name)(*args, billing=Billing.PAID, client=injected)
    assert not injected.closed
    owned = shared_client()
    monkeypatch.setattr(flows, "cloud_client", lambda **kwargs: owned.manager)
    await getattr(flows, name)(*args, billing=Billing.PAID)
    assert owned.closed


@pytest.mark.anyio
async def test_cloud_error_is_sanitized_and_owned_client_closed(monkeypatch):
    import httpx

    from mobilerun.errors.cloud import CloudConnectionError

    c = shared_client()
    c.workflows.flows.list.side_effect = httpx.ConnectError("secret-url")
    monkeypatch.setattr(flows, "cloud_client", lambda **kwargs: c.manager)
    with pytest.raises(CloudConnectionError) as error:
        await flows.list_flows(billing=Billing.NO_COMMERCE)
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
    assert await flows.list_executions(
        "f", limit=3, billing=Billing.NO_COMMERCE, client=client
    ) == [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    assert method.await_args_list[1].kwargs["page"] == 2
    assert method.await_args_list[0].kwargs["order_by"] == "startedAt"
    assert method.await_args_list[0].kwargs["order_by_direction"] == "desc"


FAILURE_PATHS = {
    "list_flows": "flows.list",
    "get_document": "flows.retrieve_document",
    "list_actions": "flows.actions.list",
    "get_action": "actions.retrieve",
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
@pytest.mark.parametrize("name,args,paid", CASES)
async def test_client_ownership_on_transport_failure(name, args, paid, monkeypatch):
    import httpx

    for injected in (False, True):
        c = shared_client()
        method = c.workflows
        for part in FAILURE_PATHS[name].split("."):
            method = getattr(method, part)
        method.side_effect = httpx.ConnectError("secret-url")
        from unittest.mock import Mock

        monkeypatch.setattr(flows, "cloud_client", Mock(return_value=c.manager))
        with pytest.raises(CloudAPIError) as error:
            await getattr(flows, name)(
                *args, billing=Billing.PAID, client=c if injected else None
            )
        assert "secret-url" not in str(error.value)
        assert c.closed is not injected


@pytest.mark.anyio
@pytest.mark.parametrize("bad", [None, [], "bad", 1])
async def test_malformed_verification_preserves_all_created_ids(bad):
    c = shared_client()
    c.workflows.flows.retrieve.return_value = {"data": bad}
    with pytest.raises(flows.FlowStagingError) as error:
        await flows.create_document(DOCUMENT, "key", billing=Billing.PAID, client=c)
    assert error.value.trigger_id == "id"
    assert error.value.action_ids == ("id",)
    assert error.value.flow_id == "id"


@pytest.mark.anyio
@pytest.mark.parametrize("failure_stage", [0, 1, 2, 3])
async def test_staging_failure_preserves_each_created_id(failure_stage):
    import httpx

    c = shared_client()
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
    with pytest.raises(flows.FlowStagingError) as error:
        await flows.create_document(doc, "key", billing=Billing.PAID, client=c)
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
    c = shared_client()
    await flows.validate_document({}, billing=Billing.NO_COMMERCE, client=c)
    c.workflows.flows.validate_document.assert_awaited_once_with(document={})


@pytest.mark.anyio
async def test_update_flow_devices_forwards_empty_list_and_key():
    client = sdk()
    await flows.update_flow_devices(
        "flow", [], "key", billing=Billing.PAID, client=client
    )
    client.workflows.flows.update.assert_awaited_once_with(
        "flow", device_ids=[], idempotency_key="key"
    )


def test_update_flow_devices_requires_paid_before_client():
    with pytest.raises(BillingError):
        flows.update_flow_devices("flow", [], "key", billing=Billing.NO_COMMERCE)
