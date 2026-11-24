"""Stateful, cost-free workflow boundary for deployment recovery tests."""

from copy import deepcopy
from types import SimpleNamespace

import httpx

from mobilerun.errors.cloud import CloudResourceNotFound
from mobilerun.sandbox.clients import create_document as stage_document


class SandboxCloud:
    def __init__(self, document, *, enabled=True):
        self.document = deepcopy(document)
        self.flow = dict(
            id="flow", name=document["name"], triggerId="trigger", enabled=enabled
        )
        self.bindings = ["action"]
        self.actions = {"action": {"id": "action", "catalogEntryId": "catalog"}}
        self.calls = []
        self.keys = {}
        self.fail = None
        self.source = "source"
        self.version = 1
        self.function_exists = True
        self.trigger_exists = True

    def hit(self, stage):
        self.calls.append(stage)
        if self.fail == stage:
            self.fail = None
            raise RuntimeError("lost response")

    async def list_functions(self, **kw):
        return (
            [dict(id="f", name="demo", latestVersion={"version": self.version})]
            if self.function_exists
            else []
        )

    async def list_flows(self, **kw):
        return [deepcopy(self.flow)] if self.flow else []

    async def list_actions(self, flow_id, **kw):
        return [{"actionId": a} for a in self.bindings]

    async def list_action_definitions(self, **kw):
        return deepcopy(list(self.actions.values()))

    async def get_action(self, action_id, **kw):
        if action_id not in self.actions:
            raise CloudResourceNotFound(operation="Get action", status=404)
        return self.actions[action_id]

    async def get_document(self, flow_id, **kw):
        self.hit("read_document")
        return deepcopy(self.document)

    async def get_version(self, function_id, version, **kw):
        return {"version": version, "status": "ready"}

    async def get_source(self, *args, **kw):
        return self.source

    async def create_version(self, function_id, source, key, **kw):
        self.source = source
        self.version = 2
        return {"version": 2}

    async def validate_document(self, document, **kw):
        return {"valid": True}

    async def disable_flow(self, flow_id, **kw):
        self.flow["enabled"] = False
        self.hit("disable")
        return deepcopy(self.flow)

    async def enable_flow(self, flow_id, **kw):
        self.flow["enabled"] = True
        self.hit("enable")
        return deepcopy(self.flow)

    async def create_action(self, params, key, **kw):
        action_id = self.keys.setdefault(key, "new" + str(len(self.keys)))
        self.actions[action_id] = dict(id=action_id, catalogEntryId="catalog", **params)
        self.hit("create")
        return {"id": action_id}

    async def replace_flow_actions(self, flow_id, refs, key, **kw):
        self.bindings = [r["action_id"] for r in refs]
        self.document["steps"] = [
            dict(
                id=r["key"],
                name=self.actions[r["action_id"]]["name"],
                uses="functions.run",
                **{"with": self.actions[r["action_id"]]["params"]},
            )
            for r in refs
        ]
        self.hit("replace")

    async def update_trigger(self, trigger_id, params, key, **kw):
        if params["activation"] == "schedule":
            on = {
                "schedule": {
                    "cron": params["schedule_rule"]["expression"],
                    "timezone": params["timezone"],
                }
            }
        else:
            on = {
                "custom": (
                    {"inputs": params["custom_payload_schema"]}
                    if params["custom_payload_schema"] is not None
                    else {}
                )
            }
        self.document["on"] = on
        self.hit("trigger")

    async def update_flow_devices(self, flow_id, devices, key, **kw):
        self.document["devices"] = devices
        self.hit("devices")

    async def delete_action(self, action_id, **kw):
        if action_id not in self.actions:
            raise CloudResourceNotFound(operation="Delete action", status=404)
        self.actions.pop(action_id)
        self.hit("delete_action")

    async def delete_flow(self, flow_id, **kw):
        if self.flow is None:
            raise CloudResourceNotFound(operation="Delete flow", status=404)
        self.flow = None
        self.bindings = []
        self.hit("delete_flow")

    async def delete_trigger(self, trigger_id, **kw):
        if not self.trigger_exists:
            raise CloudResourceNotFound(operation="Delete trigger", status=404)
        self.trigger_exists = False
        self.hit("delete_trigger")

    async def delete_function(self, function_id, **kw):
        if not self.function_exists:
            raise CloudResourceNotFound(operation="Delete function", status=404)
        self.function_exists = False
        self.hit("delete_function")


class ProcessInterrupted(BaseException):
    """Simulate termination without running Exception recovery handlers."""


class InitialSandboxCloud(SandboxCloud):
    """Real compound staging boundary backed by idempotent in-memory SDK calls."""

    def __init__(self, document):
        super().__init__(document, enabled=False)
        self.flow = None
        self.bindings = []
        self.actions = {}
        self.function_exists = False
        self.trigger_exists = False
        self.created = {}
        self.create_calls = []
        self.interrupt = None
        self.sdk = SimpleNamespace(
            workflows=SimpleNamespace(
                action_catalog=SimpleNamespace(list=self.catalog),
                triggers=SimpleNamespace(create=self.create_trigger),
                actions=SimpleNamespace(create=self.sdk_action),
                flows=SimpleNamespace(
                    create=self.sdk_flow, retrieve=self.retrieve_flow
                ),
            )
        )

    async def catalog(self, **kwargs):
        return {
            "data": {
                "items": [{"id": "catalog", "stepName": "functions.run"}],
                "pagination": {"hasNext": False},
            }
        }

    def created_response(self, stage, key, response):
        self.create_calls.append((stage, key))
        result = self.created.setdefault((stage, key), response)
        if self.interrupt == stage:
            self.interrupt = None
            raise ProcessInterrupted()
        if self.fail == stage:
            self.fail = None
            raise httpx.ConnectError("lost response")
        return result

    async def create_function(self, name, source, key, **kwargs):
        if ("function", key) not in self.created:
            self.function_exists = True
            self.source = source
        return self.created_response(
            "function", key, {"id": "f", "latestVersion": {"version": 1}}
        )

    async def create_trigger(self, *, idempotency_key, **kwargs):
        if ("trigger", idempotency_key) not in self.created:
            self.trigger_exists = True
        return self.created_response(
            "trigger", idempotency_key, {"data": {"id": "trigger"}}
        )

    async def sdk_action(self, *, idempotency_key, **kwargs):
        if ("action", idempotency_key) not in self.created:
            self.actions["action"] = {
                "id": "action",
                "catalogEntryId": "catalog",
                **kwargs,
            }
        return self.created_response(
            "action", idempotency_key, {"data": {"id": "action"}}
        )

    async def sdk_flow(self, *, idempotency_key, **kwargs):
        if ("flow", idempotency_key) not in self.created:
            self.flow = {
                "id": "flow",
                "name": kwargs["name"],
                "enabled": kwargs["enabled"],
                "triggerId": kwargs["trigger_id"],
            }
            self.bindings = [a["action_id"] for a in kwargs["actions"]]
        return self.created_response("flow", idempotency_key, {"data": {"id": "flow"}})

    async def retrieve_flow(self, flow_id):
        return {"data": deepcopy(self.flow)}

    async def create_document(self, document, key, *, billing):
        self.document = deepcopy(document)
        return await stage_document(document, key, billing=billing, client=self.sdk)
