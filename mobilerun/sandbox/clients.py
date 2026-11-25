"""Guarded async sandbox cloud function and workflow operations."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, nullcontext
from typing import Any

import httpx
from mobilerun_sdk import APIError, AsyncMobilerun
from mobilerun_sdk.types.workflows.flow_create_params import Action

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.cloud.client import cloud_client
from mobilerun.clients.cloud.errors import cloud_error
from mobilerun.clients.cloud.signed_http import _signed_client
from mobilerun.models.billing import Billing
from mobilerun.sandbox.errors import FlowCapabilityError, FlowStagingError

JSON = dict[str, Any]


def _data(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    return value


@asynccontextmanager
async def _sdk(
    billing: Billing,
    client: AsyncMobilerun | None,
    operation: str = "Cloud function request",
) -> AsyncIterator[AsyncMobilerun]:
    async with (
        nullcontext(client) if client is not None else cloud_client(billing=billing)
    ) as sdk:
        try:
            yield sdk
        except (APIError, httpx.HTTPError) as exc:
            raise cloud_error(exc, operation) from None


async def _paginate(
    method: Callable[..., Awaitable[Any]],
    *args: Any,
    limit: int | None = None,
    omit_none: bool = False,
    **kwargs: Any,
) -> list[JSON]:
    if limit is not None and limit <= 0:
        return []
    result: list[JSON] = []
    page = 1
    while True:
        data = _data(await method(*args, page=page, page_size=100, **kwargs))
        result.extend(
            item for item in data["items"] if not omit_none or item is not None
        )
        if not data["pagination"]["hasNext"] or (
            limit is not None and len(result) >= limit
        ):
            return result if limit is None else result[:limit]
        page += 1


@requires(Billing.NO_COMMERCE)
async def list_functions(
    *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """List all functions as JSON-compatible records."""
    async with _sdk(billing, client) as sdk:
        return await _paginate(sdk.compute.functions.list, omit_none=True)


@requires(Billing.NO_COMMERCE)
async def list_versions(
    function_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """List all extant versions of a function."""
    async with _sdk(billing, client) as sdk:
        return await _paginate(
            sdk.compute.functions.versions.list, function_id, omit_none=True
        )


@requires(Billing.NO_COMMERCE)
async def get_version(
    function_id: str,
    version: int,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Retrieve a function version."""
    async with _sdk(billing, client) as sdk:
        return _data(
            await sdk.compute.functions.versions.retrieve(
                version, function_id=function_id
            )
        )


@requires(Billing.NO_COMMERCE)
async def get_source(
    function_id: str,
    version: int,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> str:
    """Retrieve the source of a function version."""
    async with _sdk(billing, client) as sdk:
        return _data(await sdk.compute.functions.source(function_id, version=version))[
            "source"
        ]


@requires(Billing.PAID)
async def create_function(
    name: str,
    source: str,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Create a function and its initial version."""
    async with _sdk(billing, client) as sdk:
        return _data(
            await sdk.compute.functions.create(
                name=name, source=source, idempotency_key=idempotency_key
            )
        )


@requires(Billing.PAID)
async def create_version(
    function_id: str,
    source: str,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Create a version of a function."""
    async with _sdk(billing, client) as sdk:
        return _data(
            await sdk.compute.functions.versions.create(
                function_id, source=source, idempotency_key=idempotency_key
            )
        )


@requires(Billing.NO_COMMERCE)
async def delete_function(
    function_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> None:
    """Delete a function."""
    async with _sdk(billing, client) as sdk:
        await sdk.compute.functions.delete(function_id)


@requires(Billing.NO_COMMERCE)
async def run_logs(
    run_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> str:
    """Download run logs without sending SDK credentials to the signed URL."""
    async with _sdk(billing, client) as sdk:
        url = _data(await sdk.compute.runs.logs(run_id))["url"]
    try:
        async with _signed_client(
            httpx.AsyncClient(follow_redirects=True, timeout=30, trust_env=False)
        ) as downloader:
            response = await downloader.get(url)
            response.raise_for_status()
            return response.text
    except (
        httpx.HTTPError,
        httpx.InvalidURL,
        httpx.CookieConflict,
        httpx.StreamError,
        TypeError,
    ):
        # The exception and response may include the signed URL or arbitrary headers.
        raise cloud_error(
            httpx.RequestError("Log download failed"), "Download run logs"
        ) from None


async def _catalog_entry_id(sdk: AsyncMobilerun, step_name: str) -> str:
    """Resolve the unique catalog entry for step_name: workflow actions are created by ID."""
    entries = await _paginate(sdk.workflows.action_catalog.list)
    matches = [item["id"] for item in entries if item.get("stepName") == step_name]
    if len(matches) != 1:
        label = (
            step_name
            if step_name in ("functions.run", "webhooks.send")
            else "sandbox step"
        )
        count = "no entries" if not matches else "more than one entry"
        message = f"Workflow action catalog: found {count} for {label}"
        error = FlowCapabilityError(message, operation=message, code="request_failed")
        error.value_safe = True  # Both interpolated labels are allowlisted above.
        raise error
    return matches[0]


@requires(Billing.NO_COMMERCE)
async def disable_flow(
    flow_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Disable a flow without activating it."""
    async with _sdk(billing, client, "Cloud workflow request") as sdk:
        try:
            return _data(await sdk.workflows.flows.update(flow_id, enabled=False))
        except (APIError, httpx.HTTPError) as exc:
            raise cloud_error(exc, "Disable flow") from None


@requires(Billing.PAID)
async def enable_flow(
    flow_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Enable a flow."""
    async with _sdk(billing, client, "Cloud workflow request") as sdk:
        try:
            return _data(await sdk.workflows.flows.update(flow_id, enabled=True))
        except (APIError, httpx.HTTPError) as exc:
            raise cloud_error(exc, "Enable flow") from None


def _document(document: JSON) -> tuple[JSON, list[JSON], JSON]:
    """Validate the supported playground document before any SDK calls."""
    try:
        if not (
            set(document) <= {"version", "name", "on", "settings", "steps", "devices"}
        ):
            raise ValueError
        if not (document["version"] == 1 and isinstance(document["name"], str)):
            raise ValueError
        if document.get("settings", {}) not in ({}, {"recording": {"mode": "off"}}):
            raise ValueError
        on = document["on"]
        if not (isinstance(on, dict) and len(on) == 1):
            raise ValueError
        trigger: JSON = {"name": document["name"]}
        if "custom" in on:
            custom = on["custom"]
            if not (isinstance(custom, dict) and set(custom) <= {"inputs"}):
                raise ValueError
            trigger["activation"] = "custom"
            if "inputs" in custom:
                if not (isinstance(custom["inputs"], dict)):
                    raise ValueError
                trigger["custom_payload_schema"] = custom["inputs"]
        else:
            schedule = on["schedule"]
            if not (
                isinstance(schedule, dict) and set(schedule) == {"cron", "timezone"}
            ):
                raise ValueError
            if not (all(isinstance(schedule[k], str) for k in schedule)):
                raise ValueError
            trigger.update(
                activation="schedule",
                schedule_rule={"type": "cron", "expression": schedule["cron"]},
                timezone=schedule["timezone"],
            )
        steps = document["steps"]
        if not (isinstance(steps, list) and steps):
            raise ValueError
        for step in steps:
            if not (
                isinstance(step, dict) and set(step) == {"id", "name", "uses", "with"}
            ):
                raise ValueError
            if not (
                step["uses"] in ("functions.run", "webhooks.send")
                and isinstance(step["with"], dict)
            ):
                raise ValueError
            if not (isinstance(step["id"], str) and isinstance(step["name"], str)):
                raise ValueError
        flow: JSON = {
            "name": document["name"],
            "enabled": False,
            "recording_enabled": False,
        }
        if "devices" in document:
            if not (
                isinstance(document["devices"], list)
                and all(isinstance(d, str) for d in document["devices"])
            ):
                raise ValueError
            flow["device_ids"] = document["devices"]
        return trigger, steps, flow
    except (ValueError, KeyError, TypeError):
        raise FlowCapabilityError(
            operation="Stage flow", code="request_failed"
        ) from None


@requires(Billing.NO_COMMERCE)
async def prepare_action_payloads(
    document: JSON, *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """Resolve the exact SDK action bodies before the staging intent is journaled."""
    _, steps, _ = _document(document)
    async with _sdk(billing, client, "Resolve sandbox actions") as sdk:
        catalog = {
            uses: await _catalog_entry_id(sdk, uses)
            for uses in dict.fromkeys(step["uses"] for step in steps)
        }
    return [
        dict(
            catalog_entry_id=catalog[step["uses"]],
            name=step["name"],
            params=step["with"],
        )
        for step in steps
    ]


@requires(Billing.PAID)
async def create_document(
    document: JSON,
    idempotency_key: str,
    *,
    action_payloads: list[JSON] | None = None,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Create supported inline workflow resources and verify disabled staging."""
    trigger, steps, flow = _document(document)
    trigger_id = None
    action_ids: list[str] = []
    flow_id = None
    async with _sdk(billing, client, "Cloud workflow request") as sdk:
        try:
            if action_payloads is None:
                action_payloads = await prepare_action_payloads(
                    document, billing=billing, client=sdk
                )
            if len(action_payloads) != len(steps):
                raise ValueError("Incomplete action payloads")
            trigger_id = _data(
                await sdk.workflows.triggers.create(
                    **trigger, idempotency_key=idempotency_key + ":trigger"
                )
            )["id"]
            refs: list[Action] = []
            for position, step in enumerate(steps):
                action = _data(
                    await sdk.workflows.actions.create(
                        **action_payloads[position],
                        idempotency_key=idempotency_key + f":action:{position}",
                    )
                )
                action_ids.append(action["id"])
                refs.append(
                    {"action_id": action["id"], "position": position, "key": step["id"]}
                )
            flow_id = _data(
                await sdk.workflows.flows.create(
                    **flow,
                    trigger_id=trigger_id,
                    actions=refs,
                    idempotency_key=idempotency_key + ":flow",
                )
            )["id"]
            verified = _data(await sdk.workflows.flows.retrieve(flow_id))
            if not isinstance(verified, dict) or verified.get("enabled") is not False:
                raise FlowStagingError(trigger_id, action_ids, flow_id)
            return {"flowId": flow_id, "triggerId": trigger_id, "actionIds": action_ids}
        except (APIError, httpx.HTTPError) as exc:
            if trigger_id is not None:
                raise FlowStagingError(trigger_id, action_ids, flow_id) from None
            raise cloud_error(exc, "Stage flow") from None
        except (KeyError, TypeError, ValueError):
            raise FlowStagingError(trigger_id, action_ids, flow_id) from None


async def _request(
    client: AsyncMobilerun | None,
    billing: Billing,
    method_for: Callable[[AsyncMobilerun], Callable[..., Awaitable[Any]]],
    operation: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    async with _sdk(billing, client, "Cloud workflow request") as sdk:
        try:
            method = method_for(sdk)
            return _data(await method(*args, **kwargs))
        except (APIError, httpx.HTTPError) as exc:
            raise cloud_error(exc, operation) from None


async def _flow_pages(
    client: AsyncMobilerun | None,
    billing: Billing,
    method_for: Callable[[AsyncMobilerun], Callable[..., Awaitable[Any]]],
    operation: str,
    *,
    limit: int | None = None,
    **kwargs: Any,
) -> list[JSON]:
    if limit is not None and limit <= 0:
        return []
    async with _sdk(billing, client, operation) as sdk:
        return await _paginate(method_for(sdk), limit=limit, **kwargs)


@requires(Billing.NO_COMMERCE)
async def list_flows(
    *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """List all workflows."""
    return await _flow_pages(
        client, billing, lambda sdk: sdk.workflows.flows.list, "List flows"
    )


@requires(Billing.NO_COMMERCE)
async def get_document(
    flow_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Retrieve a workflow document."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.flows.retrieve_document,
        "Get flow document",
        flow_id,
    )


@requires(Billing.NO_COMMERCE)
async def list_actions(
    flow_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """List actions attached to a workflow."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.flows.actions.list,
        "List flow actions",
        flow_id,
    )


@requires(Billing.NO_COMMERCE)
async def list_action_definitions(
    *, billing: Billing, client: AsyncMobilerun | None = None
) -> list[JSON]:
    """List every account action, including definitions unattached to a flow."""
    return await _flow_pages(
        client,
        billing,
        lambda sdk: sdk.workflows.actions.list,
        "List action definitions",
    )


@requires(Billing.NO_COMMERCE)
async def get_trigger(
    trigger_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Retrieve a trigger."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.triggers.retrieve,
        "Get trigger",
        trigger_id,
    )


@requires(Billing.NO_COMMERCE)
async def list_executions(
    flow_id: str,
    *,
    invocation_id: str | None = None,
    limit: int | None = None,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> list[JSON]:
    """List newest workflow executions, optionally filtered by invocation."""
    kwargs: JSON = {
        "flow_id": flow_id,
        "order_by": "startedAt",
        "order_by_direction": "desc",
    }
    if invocation_id is not None:
        kwargs["invocation_id"] = invocation_id
    return await _flow_pages(
        client,
        billing,
        lambda sdk: sdk.workflows.executions.list,
        "List executions",
        limit=limit,
        **kwargs,
    )


@requires(Billing.NO_COMMERCE)
async def get_execution(
    execution_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Retrieve an execution."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.executions.retrieve,
        "Get execution",
        execution_id,
    )


@requires(Billing.NO_COMMERCE)
async def validate_document(
    document: JSON, *, billing: Billing, client: AsyncMobilerun | None = None
) -> JSON:
    """Validate a workflow document using the SDK."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.flows.validate_document,
        "Validate flow document",
        document=document,
    )


@requires(Billing.PAID)
async def replace_document(
    flow_id: str,
    document: JSON,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Replace a disabled workflow and verify it remains disabled."""
    async with _sdk(billing, client, "Cloud workflow request") as sdk:
        current = await _request(
            sdk, billing, lambda sdk: sdk.workflows.flows.retrieve, "Get flow", flow_id
        )
        if current.get("enabled") is not False:
            raise FlowCapabilityError(operation="Replace flow", code="request_failed")
        result = await _request(
            sdk,
            billing,
            lambda sdk: sdk.workflows.flows.replace_document,
            "Replace flow document",
            flow_id,
            document=document,
            idempotency_key=idempotency_key,
        )
        current = await _request(
            sdk, billing, lambda sdk: sdk.workflows.flows.retrieve, "Get flow", flow_id
        )
        if current.get("enabled") is not False:
            raise FlowCapabilityError(operation="Replace flow", code="request_failed")
        return result


def _parameters(params: JSON, allowed: set[str]) -> JSON:
    if not isinstance(params, dict) or not params.keys() <= allowed:
        raise FlowCapabilityError(
            operation="Workflow parameters", code="request_failed"
        )
    return params


_TRIGGER_FIELDS = {
    "activation",
    "name",
    "conditions",
    "custom_payload_schema",
    "description",
    "event_type",
    "payload",
    "schedule_rule",
    "timezone",
}
_ACTION_FIELDS = {"catalog_entry_id", "name", "description", "params"}


@requires(Billing.PAID)
async def create_trigger(
    params: JSON,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Create a workflow trigger from SDK fields."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.triggers.create,
        "Create trigger",
        **_parameters(params, _TRIGGER_FIELDS),
        idempotency_key=idempotency_key,
    )


@requires(Billing.PAID)
async def update_trigger(
    trigger_id: str,
    params: JSON,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Update a workflow trigger from SDK fields."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.triggers.update,
        "Update trigger",
        trigger_id,
        **_parameters(params, _TRIGGER_FIELDS),
        idempotency_key=idempotency_key,
    )


@requires(Billing.PAID)
async def create_action(
    params: JSON,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Create a workflow action from SDK fields."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.actions.create,
        "Create action",
        **_parameters(params, _ACTION_FIELDS),
        idempotency_key=idempotency_key,
    )


@requires(Billing.PAID)
async def replace_flow_actions(
    flow_id: str,
    actions: list[JSON],
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> list[JSON]:
    """Replace a flow's action references."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.flows.actions.replace,
        "Replace flow actions",
        flow_id,
        actions=actions,
        idempotency_key=idempotency_key,
    )


@requires(Billing.PAID)
async def update_flow_devices(
    flow_id: str,
    device_ids: list[str],
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Update a workflow's devices; an empty list makes it device-free."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.flows.update,
        "Update flow devices",
        flow_id,
        device_ids=device_ids,
        idempotency_key=idempotency_key,
    )


@requires(Billing.PAID)
async def fire_trigger(
    trigger_id: str,
    payload: JSON,
    idempotency_key: str,
    *,
    billing: Billing,
    client: AsyncMobilerun | None = None,
) -> JSON:
    """Fire a custom trigger."""
    return await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.triggers.fire,
        "Fire trigger",
        trigger_id,
        payload=payload,
        idempotency_key=idempotency_key,
    )


@requires(Billing.NO_COMMERCE)
async def delete_flow(
    flow_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> None:
    """Delete a workflow."""
    await _request(
        client, billing, lambda sdk: sdk.workflows.flows.delete, "Delete flow", flow_id
    )


@requires(Billing.NO_COMMERCE)
async def delete_action(
    action_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> None:
    """Delete a workflow action."""
    await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.actions.delete,
        "Delete action",
        action_id,
    )


@requires(Billing.NO_COMMERCE)
async def delete_trigger(
    trigger_id: str, *, billing: Billing, client: AsyncMobilerun | None = None
) -> None:
    """Delete a workflow trigger."""
    await _request(
        client,
        billing,
        lambda sdk: sdk.workflows.triggers.delete,
        "Delete trigger",
        trigger_id,
    )
