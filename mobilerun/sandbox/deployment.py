"""Sandbox deployment document construction and reconciliation."""

import asyncio
import copy
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.errors.cloud import CloudResourceNotFound
from mobilerun.models.billing import Billing
from mobilerun.utils.paths import local_store_path

from . import clients
from .errors import (
    FlowStagingError,
    SandboxConfirmationRequired,
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
    SandboxRunError,
    SandboxRunTimeout,
)
from .models import (
    Artifact,
    DeploymentPreview,
    DeploymentReceipt,
    Endpoint,
    OwnershipReceipt,
    resolve_billing,
    validate_device,
)
from .packaging import package
from .settings import SandboxSettings, get_sandbox_settings

JSON = dict[str, Any]


def _windows_platform() -> bool:
    return sys.platform == "win32"


@dataclass(frozen=True)
class Scope:
    """Cloud origin and API-key fingerprint; never the API key itself."""

    origin: str
    account_id: str


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_scope() -> Scope:
    """Resolve credentials exactly as the shared cloud client does."""
    key = resolve_cloud_api_key()
    if not key:
        raise SandboxDeploymentError("Sandbox credentials required")
    return Scope(str(get_settings().MOBILERUN_CLOUD_BASE_URL), _hash(key))


def stage_key(scope: Scope, name: str, digest: str, stage: str) -> str:
    """Return a stable key for the same scoped mutation intent."""
    return _hash(
        json.dumps(
            [scope.origin, scope.account_id, name, digest, stage],
            ensure_ascii=True,
            separators=(",", ":"),
        )
    )


class _FlowDumper(yaml.SafeDumper):
    pass


def _yaml_string(dumper: _FlowDumper, value: str) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style='"' if value == "on" else None
    )


_FlowDumper.add_representer(str, _yaml_string)


class ReceiptStore:
    """File-backed, scope-isolated ownership evidence for one deployment."""

    def __init__(
        self, scope: Scope, name: str, settings: SandboxSettings | None = None
    ) -> None:
        settings = settings or get_sandbox_settings()
        root = Path(
            local_store_path(
                settings.sandbox_directory,
                overridden=settings.MOBILERUN_SANDBOX_DIRECTORY is not None,
            ).path
        )
        self.scope, self.name = scope, name
        self.directory = (
            root / _hash(scope.origin + "\0" + scope.account_id) / _hash(name)
        )
        self.receipt_path = self.directory / "receipt.json"
        self.yaml_path = self.directory / "flow.yaml"

    def _validate(self, value: Any) -> OwnershipReceipt:
        try:
            expected = {
                "origin",
                "account_id",
                "name",
                "function_id",
                "flow_id",
                "trigger_id",
                "action_ids",
                "digest",
                "stages",
            }
            if not isinstance(value, dict) or set(value) != expected:
                raise ValueError
            if any(
                not isinstance(value[k], str) or not value[k]
                for k in ("origin", "account_id", "name")
            ):
                raise ValueError
            for key in ("function_id", "flow_id", "trigger_id", "digest"):
                if value[key] is not None and (
                    not isinstance(value[key], str) or not value[key]
                ):
                    raise ValueError
            for key in ("action_ids", "stages"):
                if not isinstance(value[key], list) or any(
                    not isinstance(x, str) or not x for x in value[key]
                ):
                    raise ValueError
            if (
                value["origin"] != self.scope.origin
                or value["account_id"] != self.scope.account_id
                or value["name"] != self.name
            ):
                raise SandboxDeploymentConflict("Sandbox receipt scope mismatch")
            return OwnershipReceipt(
                value["origin"],
                value["account_id"],
                value["name"],
                value["function_id"],
                value["flow_id"],
                value["trigger_id"],
                tuple(value["action_ids"]),
                value["digest"],
                tuple(value["stages"]),
            )
        except (KeyError, TypeError, ValueError):
            raise SandboxDeploymentError("Invalid sandbox receipt") from None

    def read(self) -> OwnershipReceipt | None:
        """Read and validate evidence, without creating the store."""
        try:
            return self._validate(
                json.loads(self.receipt_path.read_text(encoding="utf-8"))
            )
        except FileNotFoundError:
            return None
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise SandboxDeploymentError("Cannot read sandbox receipt") from None

    def _write_atomic(self, path: Path, text: str) -> None:
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.directory,
                prefix=".sandbox-",
                delete=False,
            ) as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
                temporary = handle.name
            os.replace(temporary, path)
        except (OSError, UnicodeError):
            raise SandboxDeploymentError("Cannot write sandbox store") from None

    def write(self, receipt: OwnershipReceipt) -> None:
        """Atomically persist a verified receipt."""
        self._validate(receipt.to_dict())
        self._write_atomic(
            self.receipt_path, json.dumps(receipt.to_dict(), ensure_ascii=True) + "\n"
        )

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold a nonblocking process-safe identity lock during deployment."""
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handle = (self.directory / ".lock").open("a+b")
        except OSError:
            raise SandboxDeploymentError("Cannot open sandbox lock") from None
        try:
            if _windows_platform():
                import msvcrt

                handle.seek(0)
                if not handle.read(1):
                    handle.write(b"\0")
                    handle.flush()
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                except OSError:
                    raise SandboxDeploymentConflict(
                        "Sandbox deployment already locked"
                    ) from None
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise SandboxDeploymentConflict(
                        "Sandbox deployment already locked"
                    ) from None
                yield
        finally:
            handle.close()

    def record_stage(
        self, receipt: OwnershipReceipt, stage: str, idempotency_key: str
    ) -> OwnershipReceipt:
        """Persist a mutation intent and its key before making the request."""
        if not stage or not idempotency_key:
            raise SandboxDeploymentError("Invalid sandbox stage")
        entry = json.dumps(
            {"stage": stage, "idempotency_key": idempotency_key},
            sort_keys=True,
            separators=(",", ":"),
        )
        result = replace(receipt, stages=receipt.stages + (entry,))
        self.write(result)
        return result

    def record_ids(self, receipt: OwnershipReceipt, **fields: Any) -> OwnershipReceipt:
        """Persist known IDs and digest after successful creation."""
        if not fields.keys() <= {
            "function_id",
            "flow_id",
            "trigger_id",
            "action_ids",
            "digest",
        }:
            raise SandboxDeploymentError("Invalid sandbox receipt update")
        result = replace(receipt, **fields)
        self.write(result)
        return result

    def write_yaml(self, document: JSON) -> None:
        """Export the authored flow document, quoting YAML's ambiguous on key."""
        self._write_atomic(
            self.yaml_path,
            yaml.dump(
                document, Dumper=_FlowDumper, sort_keys=False, allow_unicode=True
            ),
        )

    def remove_yaml(self, *, cleanup_confirmed: bool) -> None:
        """Remove exported YAML only after owned-flow cleanup is confirmed."""
        if not cleanup_confirmed:
            raise SandboxDeploymentError("Sandbox cleanup not confirmed")
        try:
            self.yaml_path.unlink(missing_ok=True)
        except OSError:
            raise SandboxDeploymentError(
                "Cannot remove sandbox flow document"
            ) from None


def build_document(
    artifact: Artifact, function_id: str, version: int, *, device: str | None
) -> JSON | None:
    """Build the authored flow document for a packaged function."""
    decl = artifact.declaration
    decl.validate()
    if decl.schedule is None and not decl.has_endpoint:
        return None
    settings = get_sandbox_settings()
    on = (
        {"schedule": {"cron": decl.schedule, "timezone": "UTC"}}
        if decl.schedule
        else {"custom": {"inputs": artifact.input_schema}}
    )
    result: JSON = {
        "version": 1,
        "name": decl.name + settings.MOBILERUN_SANDBOX_FLOW_SUFFIX,
        "on": on,
        "settings": {"recording": {"mode": "off"}},
        "steps": [
            {
                "id": "function",
                "name": decl.name + settings.MOBILERUN_SANDBOX_ACTION_SUFFIX,
                "uses": "functions.run",
                "with": {
                    "functionId": function_id,
                    "version": version,
                    "timeoutS": settings.MOBILERUN_SANDBOX_FUNCTION_TIMEOUT,
                    "input": {} if decl.schedule else "{{event.payload}}",
                },
            }
        ],
    }
    if device is not None:
        result["devices"] = [device]
    return result


def equivalent_document(actual: JSON, desired: JSON) -> bool:
    """Compare authored fields while detecting removed device/schema settings."""

    def compare(a: Any, d: Any) -> bool:
        if isinstance(d, dict):
            if not isinstance(a, dict):
                return False
            if "on" in d and "devices" not in d and a.get("devices"):
                return False
            if (
                "on" in d
                and d["on"].get("custom") == {}
                and a.get("on", {}).get("custom", {}).get("inputs") is not None
            ):
                return False
            return all(
                k in a
                and (
                    a[k] == v
                    if k in ("on", "devices", "with", "recording")
                    else compare(a[k], v)
                )
                for k, v in d.items()
            )
        if isinstance(d, list):
            return (
                isinstance(a, list)
                and len(a) == len(d)
                and all(compare(x, y) for x, y in zip(a, d, strict=True))
            )
        return a == d

    return compare(actual, desired)


@requires(Billing.PAID)
def _require_paid(*, billing: Billing) -> None:
    """Check commerce authorization before creating local or remote resources."""


def _schema_children(schema: JSON) -> Iterator[Any]:
    """Visit schema positions, never instance annotations such as default/const."""
    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependencies",
    ):
        children = schema.get(key)
        if isinstance(children, dict):
            yield from children.values()
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = schema.get(key)
        if isinstance(children, list):
            yield from children
    for key in (
        "items",
        "additionalProperties",
        "additionalItems",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
        "unevaluatedItems",
        "unevaluatedProperties",
    ):
        child = schema.get(key)
        if isinstance(child, list) and key == "items":
            yield from child
        elif isinstance(child, (dict, bool)):
            yield child


def _has_ref(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")) or any(
        _has_ref(child) for child in _schema_children(value)
    )


def _identifier(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in value)
    ):
        raise SandboxDeploymentError("Invalid sandbox resource ID")
    return value


def _version(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SandboxDeploymentError("Invalid sandbox version")
    return value


async def _wait_ready(
    function_id: str, version: int, billing: Billing, receipt: OwnershipReceipt
) -> None:
    deadline = time.monotonic() + 120.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        try:
            result = await asyncio.wait_for(
                clients.get_version(function_id, version, billing=billing), remaining
            )
        except TimeoutError:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            ) from None
        if time.monotonic() >= deadline:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        if result.get("version") != version:
            raise SandboxDeploymentError(
                "Invalid sandbox version response", receipt=receipt
            )
        status = result.get("status")
        if status == "ready":
            return
        if status == "rejected":
            raise SandboxDeploymentError("Sandbox version rejected", receipt=receipt)
        if status not in {"pending", "pending_inspection", "inspecting", "processing"}:
            raise SandboxDeploymentError(
                "Invalid sandbox version status", receipt=receipt
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SandboxDeploymentError(
                "Sandbox version inspection timed out", receipt=receipt
            )
        await asyncio.sleep(
            min(get_sandbox_settings().MOBILERUN_SANDBOX_POLL_INTERVAL, remaining)
        )


async def _delete_verified(operation: Awaitable[Any]) -> None:
    """A typed server 404 confirms that a receipt-proven resource is absent."""
    try:
        await operation
    except CloudResourceNotFound:
        pass


async def deploy_file(
    path: Path,
    *,
    device: str | None = None,
    device_free: bool = False,
    billing: Billing | str | None = None,
    yes: bool = False,
    confirm: Callable[[DeploymentPreview], bool] | None = None,
    artifact: Artifact | None = None,
) -> DeploymentReceipt:
    """Deploy a new function and optionally stage a first owned flow."""
    validate_device(device)
    if artifact is None:
        artifact = package(path)
    decl = artifact.declaration
    if device is not None and device_free:
        raise SandboxDeclarationError("Choose a device or device-free, not both")
    selected = None if device_free else device if device is not None else decl.device
    if decl.schedule is not None and selected is None and not device_free:
        raise SandboxDeclarationError(
            "Select a device or explicitly choose device-free"
        )
    # Validate declaration before any cloud calls or local writes.
    build_document(artifact, "placeholder", 1, device=selected)
    if decl.has_endpoint and _has_ref(artifact.input_schema):
        raise SandboxDeclarationError("Endpoint schema contains unsupported $ref")
    mode = resolve_billing(billing)
    preview = DeploymentPreview(decl.name, selected, decl.schedule, mode)
    if not yes and (confirm is None or confirm(preview) is not True):
        raise SandboxConfirmationRequired("Sandbox deployment requires confirmation")
    _require_paid(billing=mode)
    scope = current_scope()
    store = ReceiptStore(scope, decl.name)
    with store.lock():
        receipt = store.read()
        initial = _initial(receipt) if receipt else None
        if initial and initial.get("creation_resolved"):
            raise SandboxDeploymentConflict(
                "Finish sandbox cleanup before deploying", receipt=receipt
            )
        if initial and (
            initial["digest"] != artifact.digest or initial["device"] != selected
        ):
            raise SandboxDeploymentConflict(
                "Finish the original initial deployment first", receipt=receipt
            )
        matches = [
            f
            for f in await clients.list_functions(billing=mode)
            if f.get("name") == decl.name
        ]
        if len(matches) > 1:
            raise SandboxDeploymentConflict(
                "Multiple sandbox functions match", receipt=receipt
            )
        found = matches[0] if matches else None
        if (
            found is not None
            and (receipt is None or receipt.function_id != found.get("id"))
            and not (
                initial and initial["function_pending"] and receipt.function_id is None
            )
        ):
            raise SandboxDeploymentConflict(
                "Sandbox function is not owned", receipt=receipt
            )
        flow_name = decl.name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
        all_flows = await clients.list_flows(billing=mode)
        flows = [f for f in all_flows if f.get("name") == flow_name]
        if initial:
            if flows and not initial["flow_pending"]:
                raise SandboxDeploymentConflict(
                    "Sandbox flow is not owned", receipt=receipt
                )
            return await _first_deploy(
                store, receipt, artifact, selected, found, mode, flows
            )
        if receipt and receipt.flow_id:
            return await _redeploy(
                store, receipt, artifact, selected, found, all_flows, mode
            )
        if flows:
            raise SandboxDeploymentConflict(
                "Sandbox flow is not owned", receipt=receipt
            )
        if receipt is None:
            receipt = OwnershipReceipt(
                scope.origin,
                scope.account_id,
                decl.name,
                None,
                None,
                None,
                (),
                None,
                (),
            )
        return await _first_deploy(
            store, receipt, artifact, selected, found, mode, flows
        )


async def _first_deploy(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    selected: str | None,
    found: JSON | None,
    mode: Billing,
    visible_flows: list[JSON],
) -> DeploymentReceipt:
    scope, decl = store.scope, artifact.declaration
    state = _initial(receipt)
    if state is None:
        if found is None and receipt.function_id:
            raise SandboxDeploymentConflict(
                "Owned sandbox function is missing", receipt=receipt
            )
        # Completed cleanup starts a new intent, not a replay of deleted creations.
        # Journal position is persisted through the keys below before any request.
        intent_digest = _hash(
            json.dumps([artifact.digest, selected, len(receipt.stages)])
        )
        state = dict(
            stage="initial",
            digest=artifact.digest,
            device=selected,
            function_pending=found is None,
            flow_pending=False,
            function_key=stage_key(scope, decl.name, intent_digest, "function:create"),
            flow_key=stage_key(scope, decl.name, intent_digest, "flow:create"),
            version=None,
            document=None,
            status="created" if found is None else "unchanged",
            complete=False,
        )
        receipt = _checkpoint(store, receipt, state)
    if state["function_pending"]:
        key = state["function_key"]
        receipt = store.record_stage(receipt, "function:create", key)
        try:
            result = await clients.create_function(
                decl.name, artifact.source, key, billing=mode
            )
        except Exception:
            raise SandboxDeploymentError(
                "Sandbox function creation outcome uncertain; retry the original deployment",
                receipt=receipt,
            ) from None
        function_id = _identifier(result.get("id"))
        if receipt.function_id and receipt.function_id != function_id:
            raise SandboxDeploymentConflict(
                "Idempotent function response conflicts with remote state",
                receipt=receipt,
            )
        receipt = store.record_ids(receipt, function_id=function_id)
        version = _version((result.get("latestVersion") or {}).get("version"))
        state.update(function_pending=False, version=version)
        receipt = _checkpoint(store, receipt, state)
        if found and found.get("id") != function_id:
            raise SandboxDeploymentConflict(
                "Idempotent function response conflicts with remote state",
                receipt=receipt,
            )
    elif state["version"] is None:
        if found is None:
            raise SandboxDeploymentConflict(
                "Owned sandbox function is missing", receipt=receipt
            )
        function_id = _identifier(found.get("id"))
        version = _version((found.get("latestVersion") or {}).get("version"))
        await _wait_ready(function_id, version, mode, receipt)
        source = await clients.get_source(function_id, version, billing=mode)
        if source == artifact.source:
            status = "unchanged"
        else:
            key = stage_key(scope, decl.name, artifact.digest, "version:create")
            receipt = store.record_stage(receipt, "version:create", key)
            result = await clients.create_version(
                function_id, artifact.source, key, billing=mode
            )
            version = _version(result.get("version"))
            status = "pushed"
        state.update(version=version, status=status)
        receipt = _checkpoint(store, receipt, state)
    function_id = _identifier(receipt.function_id)
    version = _version(state["version"])
    status = state["status"]
    await _wait_ready(function_id, version, mode, receipt)
    receipt = store.record_ids(receipt, digest=artifact.digest)
    document = build_document(artifact, function_id, version, device=selected)
    endpoint = None
    if document is not None:
        if state["flow_pending"] and state["document"] != document:
            raise SandboxDeploymentConflict(
                "Initial staging intent changed", receipt=receipt
            )
        validation = await clients.validate_document(document, billing=mode)
        if validation.get("valid") is not True:
            raise SandboxDeploymentError(
                "Sandbox flow validation failed", receipt=receipt
            )
        key = state["flow_key"]
        state.update(flow_pending=True, document=document)
        receipt = _checkpoint(store, receipt, state)
        receipt = store.record_stage(receipt, "flow:create", key)
        try:
            staged = await clients.create_document(document, key, billing=mode)
        except FlowStagingError as exc:
            receipt = store.record_ids(
                receipt,
                flow_id=exc.flow_id or receipt.flow_id,
                trigger_id=exc.trigger_id or receipt.trigger_id,
                action_ids=tuple(dict.fromkeys([*receipt.action_ids, *exc.action_ids])),
            )
            raise SandboxDeploymentError(
                "Sandbox flow staging failed", receipt=receipt
            ) from None
        except Exception:
            raise SandboxDeploymentError(
                "Sandbox flow creation outcome uncertain; retry the original deployment",
                receipt=receipt,
            ) from None
        if not isinstance(staged, dict):
            raise SandboxDeploymentError(
                "Invalid sandbox flow response", receipt=receipt
            )
        flow_id = _identifier(staged.get("flowId"))
        if receipt.flow_id and receipt.flow_id != flow_id:
            raise SandboxDeploymentConflict(
                "Idempotent flow response conflicts with remote state", receipt=receipt
            )
        receipt = store.record_ids(receipt, flow_id=flow_id)
        trigger_id = _identifier(staged.get("triggerId"))
        receipt = store.record_ids(receipt, trigger_id=trigger_id)
        actions = staged.get("actionIds")
        if not isinstance(actions, list) or not actions:
            raise SandboxDeploymentError(
                "Invalid sandbox flow response", receipt=receipt
            )
        action_ids: tuple[str, ...] = ()
        for action in actions:
            action_ids += (_identifier(action),)
            receipt = store.record_ids(receipt, action_ids=action_ids)
        if any(f.get("id") != flow_id for f in visible_flows):
            raise SandboxDeploymentConflict(
                "Idempotent flow response conflicts with remote state", receipt=receipt
            )
        store.write_yaml(document)
        state.update(flow_pending=False, complete=True)
        receipt = _checkpoint(store, receipt, state)
        key = stage_key(scope, decl.name, artifact.digest, "flow:enable")
        receipt = store.record_stage(receipt, "flow:enable", key)
        try:
            enabled = await clients.enable_flow(flow_id, billing=mode)
            if not isinstance(enabled, dict) or enabled.get("enabled") is not True:
                raise ValueError("Enable not confirmed")
        except Exception:
            raise SandboxDeploymentError(
                "Sandbox flow activation failed or uncertain", receipt=receipt
            ) from None
        if decl.has_endpoint:
            endpoint = Endpoint(
                scope.origin.rstrip("/") + "/triggers/" + trigger_id + "/fire",
                trigger_id,
            )
    else:
        state["complete"] = True
        receipt = _checkpoint(store, receipt, state)
    return DeploymentReceipt(
        decl.name,
        function_id,
        version,
        status,
        receipt.flow_id,
        receipt.trigger_id,
        receipt.action_ids,
        artifact.digest,
        endpoint,
        (),
    )


def _initial(receipt: OwnershipReceipt) -> JSON | None:
    for entry in reversed(receipt.stages):
        value = json.loads(entry)
        if value.get("stage") == "initial":
            return value if not value.get("complete") else None
    return None


def _cutover(receipt: OwnershipReceipt) -> JSON | None:
    for entry in reversed(receipt.stages):
        value = json.loads(entry)
        if value.get("stage") == "cutover":
            return value if not value.get("complete") else None
    return None


def _checkpoint(
    store: ReceiptStore, receipt: OwnershipReceipt, state: JSON
) -> OwnershipReceipt:
    result = replace(
        receipt, stages=receipt.stages + (json.dumps(state, sort_keys=True),)
    )
    store.write(result)
    return result


async def _flow(flow_id: str, billing: Billing) -> JSON:
    matches = [
        f for f in await clients.list_flows(billing=billing) if f.get("id") == flow_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("enabled"), bool):
        raise SandboxDeploymentConflict("Sandbox flow state is uncertain")
    return matches[0]


async def _unshared(receipt: OwnershipReceipt, billing: Billing) -> None:
    for flow in await clients.list_flows(billing=billing):
        if flow.get("id") == receipt.flow_id:
            if flow.get("triggerId") != receipt.trigger_id:
                raise SandboxDeploymentConflict(
                    "Sandbox trigger binding changed", receipt=receipt
                )
            continue
        if receipt.trigger_id and flow.get("triggerId") == receipt.trigger_id:
            raise SandboxDeploymentConflict(
                "Sandbox trigger is shared", receipt=receipt
            )
        bindings = await clients.list_actions(
            _identifier(flow.get("id")), billing=billing
        )
        if any(a.get("actionId") in receipt.action_ids for a in bindings):
            raise SandboxDeploymentConflict("Sandbox action is shared", receipt=receipt)


def _deployment(
    receipt: OwnershipReceipt,
    artifact: Artifact,
    version: int,
    status: Literal["created", "pushed", "unchanged"],
    warnings: tuple[str, ...] = (),
) -> DeploymentReceipt:
    endpoint = None
    if artifact.declaration.has_endpoint:
        trigger_id = _identifier(receipt.trigger_id)
        endpoint = Endpoint(
            receipt.origin.rstrip("/") + "/triggers/" + trigger_id + "/fire",
            trigger_id,
        )
    return DeploymentReceipt(
        receipt.name,
        _identifier(receipt.function_id),
        version,
        status,
        receipt.flow_id,
        receipt.trigger_id,
        receipt.action_ids,
        artifact.digest,
        endpoint,
        warnings,
    )


async def _redeploy(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    selected: str | None,
    found: JSON | None,
    all_flows: list[JSON],
    mode: Billing,
) -> DeploymentReceipt:
    """Reconcile an owned flow with an on-disk write-ahead cutover journal."""
    if not found or not receipt.trigger_id or not receipt.action_ids:
        raise SandboxDeploymentConflict(
            "Sandbox ownership is incomplete", receipt=receipt
        )
    _identifier(receipt.function_id)
    _identifier(receipt.flow_id)
    _identifier(receipt.trigger_id)
    for action_id in receipt.action_ids:
        _identifier(action_id)
    flow = await _flow(receipt.flow_id, mode)
    expected_name = receipt.name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
    if (
        flow.get("name") != expected_name
        or sum(f.get("name") == expected_name for f in all_flows) != 1
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow ownership mismatch", receipt=receipt
        )
    await _unshared(receipt, mode)
    state = _cutover(receipt)
    bindings = tuple(
        _identifier(a.get("actionId"))
        for a in await clients.list_actions(receipt.flow_id, billing=mode)
    )
    allowed = (
        [tuple(state["old_ids"]), tuple(state["new_ids"])]
        if state
        else [receipt.action_ids]
    )
    if bindings not in allowed:
        raise SandboxDeploymentConflict(
            "Sandbox action bindings changed", receipt=receipt
        )
    actual = await clients.get_document(receipt.flow_id, billing=mode)
    if state:
        if state["digest"] != artifact.digest or state["device"] != selected:
            raise SandboxDeploymentConflict(
                "Finish the pending sandbox deployment first", receipt=receipt
            )
        expected = copy.deepcopy(state["before"])
        # Only the stage in flight may have either its old or new remote value.
        candidates = [expected]
        for stage, field in (
            ("replace", "steps"),
            ("trigger", "on"),
            ("devices", "devices"),
        ):
            if stage in state["done"]:
                for candidate in candidates:
                    candidate[field] = state["desired"].get(field, [])
            elif state.get("pending") == stage:
                changed = copy.deepcopy(candidates[0])
                changed[field] = state["desired"].get(field, [])
                candidates.append(changed)
        if not any(equivalent_document(actual, c) for c in candidates):
            raise SandboxDeploymentConflict(
                "Sandbox remote state diverged", receipt=receipt
            )
        version = state["version"]
        await _wait_ready(receipt.function_id, version, mode, receipt)
        desired = state["desired"]
    else:
        if store.yaml_path.exists():
            authored = yaml.safe_load(store.yaml_path.read_text(encoding="utf-8"))
            if not equivalent_document(actual, authored):
                raise SandboxDeploymentConflict(
                    "Sandbox remote document changed", receipt=receipt
                )
        version = _version((found.get("latestVersion") or {}).get("version"))
        await _wait_ready(receipt.function_id, version, mode, receipt)
        source = await clients.get_source(receipt.function_id, version, billing=mode)
        desired = build_document(
            artifact, receipt.function_id, version, device=selected
        )
        if desired is None:
            raise SandboxDeploymentConflict(
                "Remove the owned flow explicitly first", receipt=receipt
            )
        if source == artifact.source and equivalent_document(actual, desired):
            return _deployment(receipt, artifact, version, "unchanged")
        if not store.yaml_path.exists():
            raise SandboxDeploymentConflict(
                "Sandbox authored document evidence is missing", receipt=receipt
            )
        if source != artifact.source:
            key = stage_key(
                store.scope, receipt.name, artifact.digest, "version:create"
            )
            receipt = store.record_stage(receipt, "version:create", key)
            result = await clients.create_version(
                receipt.function_id, artifact.source, key, billing=mode
            )
            version = _version(result.get("version"))
            await _wait_ready(receipt.function_id, version, mode, receipt)
            desired = build_document(
                artifact, receipt.function_id, version, device=selected
            )
        if (await clients.validate_document(desired, billing=mode)).get(
            "valid"
        ) is not True:
            raise SandboxDeploymentError(
                "Sandbox flow validation failed", receipt=receipt
            )
        state = dict(
            stage="cutover",
            digest=artifact.digest,
            device=selected,
            version=version,
            before=actual,
            desired=desired,
            old_ids=list(receipt.action_ids),
            new_ids=[],
            enabled=flow["enabled"],
            done=[],
            pending=None,
            complete=False,
        )
        receipt = _checkpoint(store, receipt, state)

    def save():
        nonlocal receipt
        receipt = _checkpoint(store, receipt, state)

    def intent(stage):
        nonlocal receipt
        state["pending"] = stage
        save()
        # Includes the full desired document, so device/schema-only changes get new keys.
        digest = _hash(json.dumps([state["desired"], state["old_ids"]], sort_keys=True))
        key = stage_key(store.scope, receipt.name, digest, stage)
        receipt = store.record_stage(receipt, stage, key)
        return key

    def done(stage):
        state["done"].append(stage)
        state["pending"] = None
        save()

    try:
        # Re-prove shutdown on every recovery, including an uncertain enable.
        key = stage_key(store.scope, receipt.name, artifact.digest, "disable")
        receipt = store.record_stage(receipt, "disable", key)
        await clients.disable_flow(receipt.flow_id, billing=mode)
        if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
            raise ValueError("Shutdown not confirmed")
        if "disable" not in state["done"]:
            state["done"].append("disable")
            save()
        for index, step in enumerate(desired["steps"]):
            stage = f"create:{index}"
            if stage in state["done"]:
                continue
            old = await clients.get_action(state["old_ids"][0], billing=mode)
            catalog = _identifier(old.get("catalogEntryId"))
            key = intent(stage)
            action = await clients.create_action(
                dict(catalog_entry_id=catalog, name=step["name"], params=step["with"]),
                key,
                billing=mode,
            )
            action_id = _identifier(action.get("id"))
            state["new_ids"].append(action_id)
            receipt = store.record_ids(
                receipt,
                action_ids=tuple(dict.fromkeys([*receipt.action_ids, action_id])),
            )
            done(stage)
        if "replace" not in state["done"]:
            if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown no longer confirmed")
            key = intent("replace")
            refs = [
                dict(action_id=a, position=i, key=desired["steps"][i]["id"])
                for i, a in enumerate(state["new_ids"])
            ]
            await clients.replace_flow_actions(receipt.flow_id, refs, key, billing=mode)
            done("replace")
        if state["before"]["on"] != desired["on"] and "trigger" not in state["done"]:
            await _unshared(receipt, mode)
            on = desired["on"]
            params = dict(
                activation="custom",
                custom_payload_schema=on.get("custom", {}).get("inputs"),
                schedule_rule=None,
                timezone=None,
            )
            if "schedule" in on:
                params.update(
                    activation="schedule",
                    schedule_rule=dict(type="cron", expression=on["schedule"]["cron"]),
                    timezone=on["schedule"]["timezone"],
                )
            key = intent("trigger")
            await clients.update_trigger(receipt.trigger_id, params, key, billing=mode)
            done("trigger")
        if (
            state["before"].get("devices", []) != desired.get("devices", [])
            and "devices" not in state["done"]
        ):
            key = intent("devices")
            await clients.update_flow_devices(
                receipt.flow_id, desired.get("devices", []), key, billing=mode
            )
            done("devices")
        intent("verify")
        await _unshared(receipt, mode)
        if (await _flow(receipt.flow_id, mode))[
            "enabled"
        ] is not False or not equivalent_document(
            await clients.get_document(receipt.flow_id, billing=mode), desired
        ):
            raise ValueError("Cutover verification failed")
        verified = tuple(
            a.get("actionId")
            for a in await clients.list_actions(receipt.flow_id, billing=mode)
        )
        if verified != tuple(state["new_ids"]):
            raise ValueError("Cutover bindings changed")
        done("verify")
        unresolved = []
        for action_id in state["old_ids"]:
            stage = "delete:" + action_id
            if stage in state["done"]:
                continue
            try:
                intent(stage)
                await _unshared(receipt, mode)
                await _delete_verified(clients.delete_action(action_id, billing=mode))
                done(stage)
            except Exception:
                unresolved.append(action_id)
        if unresolved:
            raise ValueError("Old actions unresolved")
        receipt = store.record_ids(
            receipt, action_ids=tuple(state["new_ids"]), digest=artifact.digest
        )
        store.write_yaml(desired)
        if state["enabled"]:
            await _unshared(receipt, mode)
            if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown no longer confirmed")
            intent("enable")
            await clients.enable_flow(receipt.flow_id, billing=mode)
            if (await _flow(receipt.flow_id, mode))["enabled"] is not True:
                raise ValueError("Enable not confirmed")
            done("enable")
        state["complete"] = True
        save()
        warnings = (
            () if state["enabled"] else ("Flow remains disabled (enabled=False)",)
        )
        return _deployment(receipt, artifact, version, "pushed", warnings)
    except Exception:
        # Do not claim rollback. Read the actual state and expose it in durable evidence.
        try:
            key = stage_key(
                store.scope, receipt.name, artifact.digest, "failure:shutdown"
            )
            receipt = store.record_stage(receipt, "failure:shutdown", key)
            await clients.disable_flow(receipt.flow_id, billing=mode)
        except Exception:
            pass
        try:
            enabled = (await _flow(receipt.flow_id, mode))["enabled"]
        except Exception:
            enabled = "uncertain"
        state["warning"] = f"Cutover incomplete; flow enabled={enabled}"
        save()
        raise SandboxDeploymentError(state["warning"], receipt=receipt) from None


async def cleanup_owned(
    receipt: OwnershipReceipt,
    *,
    delete_function: bool = False,
    billing: Billing | str | None = None,
) -> None:
    """Best-effort deletion using local creation evidence, never resource names."""
    scope = current_scope()
    store = ReceiptStore(scope, receipt.name)
    mode = resolve_billing(billing)
    with store.lock():
        stored = store.read()
        if (
            stored != receipt
            or receipt.origin != scope.origin
            or receipt.account_id != scope.account_id
        ):
            raise SandboxDeploymentConflict(
                "Sandbox cleanup receipt mismatch", receipt=stored
            )
        unresolved = []
        initial = _initial(receipt)
        unknown_initial = (
            initial
            and not initial.get("creation_resolved")
            and (
                (initial["function_pending"] and not receipt.function_id)
                or (
                    initial["flow_pending"]
                    and (
                        not receipt.flow_id
                        or not receipt.trigger_id
                        or len(receipt.action_ids) != len(initial["document"]["steps"])
                    )
                )
            )
        )
        if unknown_initial:
            # The compound create can have unreturned trigger/action/flow IDs. Deleting
            # its known dependencies would also make the original-key replay unsafe.
            raise SandboxDeploymentError(
                "Sandbox initial resource creation outcome uncertain; retry the original deployment before cleanup",
                receipt=receipt,
            )
        if initial:
            # Preserve resolved creation evidence before cleanup clears any IDs.
            initial["creation_resolved"] = True
            receipt = _checkpoint(store, receipt, initial)
        state = _cutover(receipt)
        uncertain_creation = bool(
            state and str(state.get("pending", "")).startswith("create:")
        )
        if uncertain_creation:
            unresolved.append("action creation outcome uncertain: " + state["pending"])

        def intent(stage: str, resource_id: str) -> None:
            nonlocal receipt
            key = stage_key(scope, receipt.name, resource_id, "cleanup:" + stage)
            receipt = store.record_stage(receipt, "cleanup:" + stage, key)

        flow_deleted = receipt.flow_id is None
        if receipt.flow_id:
            try:
                # A receipt-proven resource that is already absent needs no mutation.
                present = any(
                    f.get("id") == receipt.flow_id
                    for f in await clients.list_flows(billing=mode)
                )
                if not present:
                    receipt = store.record_ids(receipt, flow_id=None)
                    flow_deleted = True
                    store.remove_yaml(cleanup_confirmed=True)
                else:
                    intent("flow:disable", receipt.flow_id)
                    await _unshared(receipt, mode)
                    bindings = {
                        a.get("actionId")
                        for a in await clients.list_actions(
                            receipt.flow_id, billing=mode
                        )
                    }
                    if not bindings <= set(receipt.action_ids):
                        raise ValueError("Foreign binding")
                    await clients.disable_flow(receipt.flow_id, billing=mode)
                    if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
                        raise ValueError("Shutdown uncertain")
                    intent("flow:delete", receipt.flow_id)
                    await _delete_verified(
                        clients.delete_flow(receipt.flow_id, billing=mode)
                    )
                    if any(
                        f.get("id") == receipt.flow_id
                        for f in await clients.list_flows(billing=mode)
                    ):
                        raise ValueError("Deletion uncertain")
                    receipt = store.record_ids(receipt, flow_id=None)
                    flow_deleted = True
                    store.remove_yaml(cleanup_confirmed=True)
            except Exception:
                unresolved.append(receipt.flow_id)
        for action_id in tuple(receipt.action_ids):
            try:
                # Include the remaining owned flow: its bindings block dependency deletion.
                for flow in await clients.list_flows(billing=mode):
                    if any(
                        a.get("actionId") == action_id
                        for a in await clients.list_actions(
                            _identifier(flow.get("id")), billing=mode
                        )
                    ):
                        raise ValueError("Action still referenced")
                intent("action:delete", action_id)
                await _delete_verified(
                    clients.delete_action(_identifier(action_id), billing=mode)
                )
                receipt = store.record_ids(
                    receipt,
                    action_ids=tuple(a for a in receipt.action_ids if a != action_id),
                )
            except Exception:
                unresolved.append(action_id)
        if receipt.trigger_id:
            try:
                if any(
                    f.get("triggerId") == receipt.trigger_id
                    for f in await clients.list_flows(billing=mode)
                ):
                    raise ValueError("Trigger still referenced")
                intent("trigger:delete", receipt.trigger_id)
                await _delete_verified(
                    clients.delete_trigger(
                        _identifier(receipt.trigger_id), billing=mode
                    )
                )
                receipt = store.record_ids(receipt, trigger_id=None)
            except Exception:
                unresolved.append(receipt.trigger_id)
        if delete_function and receipt.function_id:
            try:
                if not flow_deleted or receipt.action_ids or uncertain_creation:
                    raise ValueError("Owned dependencies remain")
                # Flow bindings omit unattached actions. Inspect the complete account
                # definition inventory before deleting a potentially shared function.
                definitions = await clients.list_action_definitions(billing=mode)
                if not isinstance(definitions, list):
                    raise ValueError("Action sharing cannot be verified")
                for action in definitions:
                    if not isinstance(action, dict):
                        raise ValueError("Action definition unavailable")
                    _identifier(action.get("id"))
                    uses = action.get("uses")
                    params = action.get("params")
                    if not isinstance(uses, str) or not uses:
                        raise ValueError("Action type unavailable")
                    if params is None and uses != "functions.run":
                        continue
                    if not isinstance(params, dict) or (
                        uses == "functions.run" and "functionId" not in params
                    ):
                        raise ValueError("Action references unavailable")
                    pending = [params]
                    while pending:
                        node = pending.pop()
                        if isinstance(node, dict):
                            if (
                                "functionId" in node
                                and _identifier(node["functionId"])
                                == receipt.function_id
                            ):
                                raise ValueError("Function shared")
                            pending.extend(node.values())
                        elif isinstance(node, list):
                            pending.extend(node)
                # Functions may be called by any action; inspect all remaining flow documents.
                for flow in await clients.list_flows(billing=mode):
                    doc = await clients.get_document(
                        _identifier(flow.get("id")), billing=mode
                    )
                    if any(
                        s.get("with", {}).get("functionId") == receipt.function_id
                        for s in doc.get("steps", [])
                    ):
                        raise ValueError("Function shared")
                intent("function:delete", receipt.function_id)
                await _delete_verified(
                    clients.delete_function(
                        _identifier(receipt.function_id), billing=mode
                    )
                )
                receipt = store.record_ids(receipt, function_id=None)
            except Exception:
                unresolved.append(receipt.function_id)
        if unresolved:
            raise SandboxDeploymentError(
                "Sandbox cleanup unresolved IDs: "
                + ", ".join(str(i) for i in unresolved),
                receipt=receipt,
            )
        if initial:
            initial["complete"] = True
            receipt = _checkpoint(store, receipt, initial)


T = TypeVar("T")


async def _remote_request(request: Awaitable[T]) -> T:
    """Never propagate transport bodies, credentials, or signed URLs in errors."""
    try:
        return await request
    except (SandboxDeploymentError, SandboxRunError):
        raise
    except Exception:
        raise SandboxRunError("Sandbox cloud request failed") from None


async def list_deployments(*, billing: Billing | str | None = None) -> list[JSON]:
    """List functions read-only, omitting credential and inspection metadata."""
    records = await _remote_request(
        clients.list_functions(billing=resolve_billing(billing))
    )
    result = []
    for record in records:
        item = {"id": _identifier(record.get("id")), "name": record.get("name")}
        latest = record.get("latestVersion")
        if isinstance(latest, dict):
            item["latestVersion"] = {
                k: latest[k] for k in ("version", "status") if k in latest
            }
        result.append(item)
    return result


async def _find_remote(name: str, mode: Billing) -> tuple[JSON, JSON | None]:
    functions = [
        f for f in await list_deployments(billing=mode) if f.get("name") == name
    ]
    if len(functions) != 1:
        raise SandboxDeploymentError(
            "Sandbox function missing or ambiguous; deploy the function first",
            missing_deployment=not functions,
        )
    flows = [
        f
        for f in await _remote_request(clients.list_flows(billing=mode))
        if f.get("name") == name + get_sandbox_settings().MOBILERUN_SANDBOX_FLOW_SUFFIX
    ]
    if len(flows) > 1:
        raise SandboxDeploymentConflict("Multiple sandbox flows match")
    flow = flows[0] if flows else None
    if flow is not None:
        _identifier(flow.get("id"))
        _identifier(flow.get("triggerId"))
    return functions[0], flow


async def _latest_ready(function_id: str, mode: Billing) -> int:
    versions = await _remote_request(clients.list_versions(function_id, billing=mode))
    ready = [
        _version(v.get("version"))
        for v in versions
        if isinstance(v, dict) and v.get("status") == "ready"
    ]
    if not ready:
        raise SandboxDeploymentError(
            "Sandbox function has no ready version; deploy a ready version first",
            missing_deployment=True,
        )
    return max(ready)


def _endpoint_schema(document: JSON) -> JSON | None:
    custom = document.get("on", {}).get("custom")
    if isinstance(custom, dict) and isinstance(custom.get("inputs"), dict):
        return custom["inputs"]
    return None


async def describe_remote(name: str, *, billing: Billing | str | None = None) -> JSON:
    """Describe current server state; neither names nor reads confer ownership."""
    mode = resolve_billing(billing)
    function, flow = await _find_remote(name, mode)
    function_id = function["id"]
    result: JSON = {
        "function_id": function_id,
        "version": await _latest_ready(function_id, mode),
        "flow_id": flow["id"] if flow else None,
    }
    if flow:
        document = await _remote_request(clients.get_document(flow["id"], billing=mode))
        if _endpoint_schema(document) is not None:
            trigger_id = flow["triggerId"]
            result["endpoint"] = {
                "url": current_scope().origin.rstrip("/")
                + "/triggers/"
                + trigger_id
                + "/fire",
                "trigger_id": trigger_id,
            }
    return result


async def _owned_remote(
    store: ReceiptStore, mode: Billing, *, activating: bool
) -> tuple[OwnershipReceipt, JSON, JSON]:
    receipt = store.read()
    function, flow = await _find_remote(store.name, mode)
    if flow is None:
        raise SandboxRunError(
            "No flow; deploy as an endpoint (default) or with a schedule first",
            missing_deployment=True,
        )
    if (
        receipt is None
        or receipt.function_id != function["id"]
        or receipt.flow_id != flow["id"]
        or receipt.trigger_id != flow["triggerId"]
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow is not receipt-owned", receipt=receipt
        )
    if activating and (_cutover(receipt) or _initial(receipt)):
        raise SandboxDeploymentConflict(
            "Sandbox cutover is incomplete; finish deployment before activation",
            receipt=receipt,
        )
    bindings = tuple(
        _identifier(a.get("actionId"))
        for a in await _remote_request(clients.list_actions(flow["id"], billing=mode))
    )
    if (
        not bindings
        or (set(bindings) - set(receipt.action_ids))
        or (not _cutover(receipt) and bindings != receipt.action_ids)
    ):
        raise SandboxDeploymentConflict(
            "Sandbox action bindings changed", receipt=receipt
        )
    await _remote_request(_unshared(receipt, mode))
    document = await _remote_request(clients.get_document(flow["id"], billing=mode))
    if activating and store.yaml_path.exists():
        try:
            authored = yaml.safe_load(store.yaml_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            raise SandboxDeploymentConflict(
                "Sandbox document evidence is unreadable", receipt=receipt
            ) from None
        if not equivalent_document(document, authored):
            raise SandboxDeploymentConflict(
                "Sandbox remote document changed", receipt=receipt
            )
    return receipt, flow, document


async def _flow_version(function_id: str, document: JSON, mode: Billing) -> JSON:
    steps = document.get("steps", [])
    if (
        not isinstance(steps, list)
        or len(steps) != 1
        or not isinstance(steps[0], dict)
        or steps[0].get("uses") != "functions.run"
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow must contain its owned function step"
        )
    params = steps[0].get("with")
    if not isinstance(params, dict) or params.get("functionId") != function_id:
        raise SandboxDeploymentConflict("Sandbox flow function binding changed")
    version = _version(params.get("version"))
    record = await _remote_request(
        clients.get_version(function_id, version, billing=mode)
    )
    if record.get("version") != version or record.get("status") != "ready":
        raise SandboxDeploymentError("Sandbox flow function version is not ready")
    return record


def _validate_payload(payload: JSON, schema: Any) -> None:
    try:
        # JSON serialization rejects non-JSON payloads before transport, including NaN.
        json.dumps(payload, allow_nan=False)
        if not isinstance(payload, dict):
            raise ValueError
        if schema is not None:
            from jsonschema.validators import validator_for

            # Never resolve remote/local references as part of validation.
            if not isinstance(schema, (dict, bool)) or _has_ref(schema):
                raise ValueError
            validator = validator_for(schema)
            validator.check_schema(schema)
            # Format assertions belong to the server; no optional format libraries.
            validator(schema).validate(payload)
    except Exception:
        raise SandboxRunError(
            "Sandbox payload does not match the input schema"
        ) from None


async def fire_remote(
    name: str, payload: JSON, *, billing: Billing | str | None = None
) -> JSON:
    """Fire an owned endpoint once, without waiting for execution creation."""
    mode = resolve_billing(billing)
    _require_paid(billing=mode)
    store = ReceiptStore(current_scope(), name)
    with store.lock():
        receipt, flow, document = await _owned_remote(store, mode, activating=True)
        on = document.get("on", {})
        if "schedule" in on:
            raise SandboxRunError(
                "Unsupported scheduled trigger; deploy as an endpoint to fire"
            )
        schema = _endpoint_schema(document)
        if schema is None:
            raise SandboxRunError("Unsupported trigger; deploy as an endpoint to fire")
        version = await _flow_version(receipt.function_id, document, mode)
        _validate_payload(payload, schema)
        if version.get("inputSchema") is not None:
            _validate_payload(payload, version["inputSchema"])
        fired = await _remote_request(
            clients.fire_trigger(
                flow["triggerId"],
                copy.deepcopy(payload),
                uuid.uuid4().hex,
                billing=mode,
            )
        )
        invocation_id = _run_identifier(fired.get("invocationId"))
        count = fired.get("enqueuedCount")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise SandboxRunError(
                "Sandbox invocation did not enqueue an execution",
                invocation_id=invocation_id,
            )
        return {"invocation_id": invocation_id}


async def recent_runs(
    name: str, limit: int = 10, *, billing: Billing | str | None = None
) -> list[JSON]:
    """List newest executions using the boundary's ordered, paginated query."""
    if limit <= 0:
        return []
    mode = resolve_billing(billing)
    _, flow = await _find_remote(name, mode)
    if flow is None:
        return []
    records = await _remote_request(
        clients.list_executions(flow["id"], limit=limit, billing=mode)
    )
    return [_snapshot(record) for record in records[:limit]]


async def set_enabled(
    name: str, enabled: bool, *, billing: Billing | str | None = None
) -> JSON:
    """Activate only owned ready flows; owned shutdown needs no commerce."""
    if not isinstance(enabled, bool):
        raise SandboxDeploymentError("Sandbox enabled must be a boolean")
    mode = resolve_billing(billing)
    if enabled:
        _require_paid(billing=mode)
    store = ReceiptStore(current_scope(), name)
    with store.lock():
        receipt, flow, document = await _owned_remote(store, mode, activating=enabled)
        if enabled:
            await _flow_version(receipt.function_id, document, mode)
        key = stage_key(
            store.scope,
            name,
            receipt.digest or "",
            "flow:enable" if enabled else "flow:disable",
        )
        store.record_stage(receipt, "flow:enable" if enabled else "flow:disable", key)
        operation = clients.enable_flow if enabled else clients.disable_flow
        await _remote_request(operation(flow["id"], billing=mode))
        actual = await _remote_request(_flow(flow["id"], mode))
        if actual["enabled"] is not enabled:
            raise SandboxDeploymentError(
                "Sandbox enabled state is unconfirmed", receipt=store.read()
            )
        return {"id": flow["id"], "enabled": actual["enabled"]}


def _snapshot(record: JSON) -> JSON:
    """Keep the execution contract, not raw server errors or signed artifacts."""
    result: JSON = {
        "id": _run_identifier(record.get("id")),
        "status": record.get("status"),
    }
    for key in ("flowId", "invocationId", "startedAt", "finishedAt", "result"):
        if key in record:
            result[key] = copy.deepcopy(record[key])
    return result


async def execution_snapshot(
    execution_id: str, *, billing: Billing | str | None = None
) -> JSON:
    """Retrieve a fresh execution snapshot, preserving partial result metadata."""
    execution_id = _run_identifier(execution_id)
    record = await _remote_request(
        clients.get_execution(execution_id, billing=resolve_billing(billing))
    )
    if record.get("id") != execution_id:
        raise SandboxRunError(
            "Sandbox execution ID mismatch", execution_id=execution_id
        )
    return _snapshot(record)


def _timeout(invocation_id: str | None, execution_id: str | None) -> SandboxRunTimeout:
    return SandboxRunTimeout(
        "Sandbox execution deadline exceeded",
        invocation_id=invocation_id,
        execution_id=execution_id,
    )


def _run_identifier(value: Any) -> str:
    try:
        return _identifier(value)
    except SandboxDeploymentError:
        raise SandboxRunError("Invalid sandbox run identifier") from None


async def _within_deadline(
    operation: Callable[[], Awaitable[T]],
    deadline: float,
    invocation_id: str | None,
    execution_id: str | None,
) -> T:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout(invocation_id, execution_id)
    try:
        result = await asyncio.wait_for(operation(), remaining)
        if time.monotonic() >= deadline:
            raise _timeout(invocation_id, execution_id)
        return result
    except TimeoutError:
        raise _timeout(invocation_id, execution_id) from None
    except SandboxRunError as error:
        error.invocation_id = error.invocation_id or invocation_id
        error.execution_id = error.execution_id or execution_id
        raise


async def _poll_pause(
    deadline: float, invocation_id: str | None, execution_id: str | None
) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _timeout(invocation_id, execution_id)
    await asyncio.sleep(
        min(get_sandbox_settings().MOBILERUN_SANDBOX_POLL_INTERVAL, remaining)
    )


async def resolve_execution(
    name: str,
    invocation_id: str,
    *,
    deadline: float,
    billing: Billing | str | None = None,
) -> str:
    """Resolve an invocation, sharing the caller's absolute monotonic deadline."""
    invocation_id = _run_identifier(invocation_id)
    if (
        not isinstance(deadline, (int, float))
        or isinstance(deadline, bool)
        or not math.isfinite(deadline)
    ):
        raise SandboxRunError(
            "Sandbox deadline must be finite", invocation_id=invocation_id
        )
    mode = resolve_billing(billing)
    while True:
        _, flow = await _within_deadline(
            lambda: _find_remote(name, mode), deadline, invocation_id, None
        )
        if flow is None:
            raise SandboxRunError(
                "No flow; deploy as an endpoint first", invocation_id=invocation_id
            )
        rows = await _within_deadline(
            lambda flow_id=flow["id"]: _remote_request(
                clients.list_executions(
                    flow_id, invocation_id=invocation_id, billing=mode
                )
            ),
            deadline,
            invocation_id,
            None,
        )
        for row in rows:
            if (
                row.get("invocationId", invocation_id) == invocation_id
                and row.get("flowId", flow["id"]) == flow["id"]
            ):
                return _run_identifier(row.get("id"))
        await _poll_pause(deadline, invocation_id, None)


def _function_steps(snapshot: JSON) -> list[JSON]:
    result = snapshot.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("steps"), list):
        return []
    return [
        s
        for s in result["steps"]
        if isinstance(s, dict) and s.get("uses") == "functions.run"
    ]


def _check_success(snapshot: JSON, invocation_id: str | None) -> None:
    steps = _function_steps(snapshot)
    execution_id = snapshot["id"]
    if not steps:
        raise SandboxRunError(
            "Sandbox execution has no function result",
            invocation_id=invocation_id,
            execution_id=execution_id,
        )
    for step in steps:
        body = step.get("body")
        if (
            step.get("status") not in (None, "success")
            or not isinstance(body, dict)
            or body.get("ok") is False
            or body.get("error") is not None
            or body.get("outcome") not in (None, "complete", "succeeded", "partial")
            or "result" not in body
        ):
            raise SandboxRunError(
                "Sandbox function body failed or is invalid",
                invocation_id=invocation_id,
                execution_id=execution_id,
            )


async def wait_result(
    name: str,
    *,
    invocation_id: str | None = None,
    execution_id: str | None = None,
    timeout: float = 300.0,
    billing: Billing | str | None = None,
) -> JSON:
    """Wait with one deadline; a timeout does not imply remote cancellation."""
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or not math.isfinite(timeout)
        or timeout < 0
    ):
        raise SandboxRunError("Sandbox timeout must be a finite nonnegative number")
    if invocation_id is None and execution_id is None:
        raise SandboxRunError("Provide an invocation or execution ID")
    if invocation_id is not None:
        _run_identifier(invocation_id)
    if execution_id is not None:
        _run_identifier(execution_id)
    deadline = time.monotonic() + timeout
    if execution_id is None:
        execution_id = await resolve_execution(
            name, invocation_id, deadline=deadline, billing=billing
        )
    while True:
        snapshot = await _within_deadline(
            lambda: execution_snapshot(execution_id, billing=billing),
            deadline,
            invocation_id,
            execution_id,
        )
        if invocation_id is not None and snapshot.get("invocationId") not in (
            None,
            invocation_id,
        ):
            raise SandboxRunError(
                "Sandbox invocation ID mismatch",
                invocation_id=invocation_id,
                execution_id=execution_id,
            )
        status = snapshot.get("status")
        if status == "success":
            _check_success(snapshot, invocation_id)
            return snapshot
        if status not in ("pending", "running"):
            label = (
                status
                if status in ("failed", "cancelled", "skipped", "invalid", "unknown")
                else "unknown"
            )
            raise SandboxRunError(
                "Sandbox execution " + label,
                invocation_id=invocation_id,
                execution_id=execution_id,
            )
        await _poll_pause(deadline, invocation_id, execution_id)


async def execution_logs(
    execution_id: str, *, billing: Billing | str | None = None
) -> str:
    """Read inline function logs, or download available compute logs safely."""
    snapshot = await execution_snapshot(execution_id, billing=billing)
    for step in _function_steps(snapshot):
        body = step.get("body")
        if not isinstance(body, dict):
            continue
        tail = "\n".join(
            body[key]
            for key in ("stdout", "stderr")
            if isinstance(body.get(key), str) and body[key]
        )
        if tail:
            return tail
        logs = body.get("logs")
        if logs == "available":
            if not body.get("runId"):
                raise SandboxRunError(
                    "Sandbox compute logs have no run ID", execution_id=execution_id
                )
            return await _remote_request(
                clients.run_logs(
                    _run_identifier(body["runId"]), billing=resolve_billing(billing)
                )
            )
        if isinstance(logs, str) and logs != "available":
            return logs
        if isinstance(logs, list):
            return "\n".join(
                part if isinstance(part, str) else json.dumps(part, ensure_ascii=True)
                for part in logs
            )
        return ""
    raise SandboxRunError(
        "Sandbox execution has no function logs", execution_id=execution_id
    )
