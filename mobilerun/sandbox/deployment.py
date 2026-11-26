"""Sandbox deployment document construction and reconciliation."""

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar

import httpx
import yaml

from mobilerun.clients.cloud.billing import requires
from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.core.webhooks import DEFAULT_WEBHOOK_EVENT, CloudWebhooks
from mobilerun.errors.cloud import CloudResourceNotFound
from mobilerun.models.billing import Billing
from mobilerun.utils.paths import local_store_path

from . import clients
from .errors import (
    FlowStagingError,
    SandboxDeclarationError,
    SandboxDeploymentConflict,
    SandboxDeploymentError,
    SandboxRunError,
    SandboxRunTimeout,
)
from .models import (
    FILE_MIME_TYPES,
    FILE_UPLOAD_USES,
    WEBHOOK_REGISTER_HINT,
    Artifact,
    DeploymentReceipt,
    Endpoint,
    OwnershipReceipt,
    is_result_path_part,
    resolve_billing,
    validate_device,
    webhook_endpoint_id,
)
from .packaging import package, verify_scheduled_artifact
from .settings import (
    WEBHOOK_STEP_ID,
    SandboxSettings,
    get_sandbox_settings,
)

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
    base = httpx.URL(str(get_settings().MOBILERUN_CLOUD_BASE_URL))
    path = base.raw_path
    base = base.copy_with(raw_path=path if path.endswith(b"/") else path + b"/")
    return Scope(str(base), _hash(key))


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
        # Keep a legacy receipt, its authored YAML, lock and retry keys together.
        if scope.origin.endswith("/") and not scope.origin.endswith("//"):
            legacy = scope.origin[:-1]
            prior = root / _hash(legacy + "\0" + scope.account_id) / _hash(name)
            if (prior / "receipt.json").exists():
                if (self.directory / "receipt.json").exists():
                    raise SandboxDeploymentConflict(
                        "Conflicting sandbox receipt directories"
                    )
                self.directory = prior
                self.scope = replace(scope, origin=legacy)
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


def _result_template(path: str = "") -> str:
    return "{{steps.function.body.result" + path + "}}"


_FileReferences = dict[tuple[str, ...], dict[str, str]]


def _webhook_result(output: JSON | None, references: _FileReferences) -> Any:
    """Project the inspected serialization schema without templating file bytes."""
    invalid_path = "Webhook files must point to required, non-null strings through supported fields"

    def project(schema: Any, path: str, pending: _FileReferences) -> Any:
        if not pending:
            return _result_template(path)
        if (
            not isinstance(schema, dict)
            or schema.get("nullable") is True
            or ({"anyOf", "oneOf"} & schema.keys())
        ):
            raise SandboxDeclarationError(invalid_path)
        if () in pending:
            if schema.get("type") != "string" or len(pending) != 1:
                raise SandboxDeclarationError(invalid_path)
            return pending[()]
        if schema.get("x-mobilerun-dynamic-serializer"):
            raise SandboxDeclarationError(
                "Webhook result branches on file paths cannot use custom or duck-typed serializers"
            )

        children: dict[str, _FileReferences] = {}
        for (part, *rest), reference in pending.items():
            children.setdefault(part, {})[tuple(rest)] = reference
        if schema.get("type") == "object":
            fields = schema.get("properties", {})
            required = set(schema.get("required", []))
            if not children.keys() <= required or not children.keys() <= fields.keys():
                raise SandboxDeclarationError(invalid_path)
            if schema.get("additionalProperties", False) is not False:
                raise SandboxDeclarationError(
                    "Webhook result objects on file paths cannot allow extra properties"
                )
            if (
                schema.get("x-mobilerun-omittable-fields")
                or not fields.keys() <= required
            ):
                raise SandboxDeclarationError(
                    "Webhook result objects on file paths must require every serialized property"
                )
            if not all(is_result_path_part(field) for field in fields):
                raise SandboxDeclarationError(
                    "Webhook result contains a field that cannot be addressed by a workflow template"
                )
            return {
                field: project(child, f"{path}.{field}", children.get(field, {}))
                for field, child in fields.items()
            }
        if schema.get("type") == "array":
            length = schema.get("minItems")
            if type(length) is not int or length != schema.get("maxItems"):
                raise SandboxDeclarationError(
                    "Webhook files require fixed-length lists on file paths; set min_length and max_length to the same value"
                )
            if not all(part.isdecimal() and int(part) < length for part in children):
                raise SandboxDeclarationError(invalid_path)
            return [
                project(schema.get("items"), f"{path}.{i}", children.get(str(i), {}))
                for i in range(length)
            ]
        raise SandboxDeclarationError(invalid_path)

    return project(output, "", references)


def build_document(
    artifact: Artifact,
    function_id: str,
    version: int,
    *,
    device: str | None,
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
                # Legacy templates resolve steps by action name, not by ID.
                "name": (
                    "function"
                    if decl.files or decl.webhook is not None
                    else decl.name + settings.MOBILERUN_SANDBOX_ACTION_SUFFIX
                ),
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
    webhook_result = _result_template()
    if decl.files and decl.files_folder is not None:
        folder = "/" + decl.files_folder.removeprefix("/")
        stored_folder = unicodedata.normalize("NFC", folder)
        result["settings"]["delivery"] = {
            "destination": "mobilerun",
            "folder": folder,
        }
        references: _FileReferences = {}
        for index, (filename, path) in enumerate(decl.files.items(), 1):
            upload_id = f"upload_{index}"
            result["steps"].append(
                {
                    "id": upload_id,
                    "name": upload_id,
                    "uses": FILE_UPLOAD_USES,
                    "with": {
                        "filename": filename,
                        "mimeType": FILE_MIME_TYPES[Path(filename).suffix.lower()],
                        "content": _result_template(f".{path}"),
                        "contentEncoding": "base64",
                    },
                }
            )
            if decl.webhook is not None:
                uploaded_path = f"{stored_folder}/{filename}"
                if "{{" in uploaded_path:
                    raise SandboxDeclarationError(
                        "Webhook file paths cannot contain template syntax"
                    )
                # A field may be uploaded under more than one name; link to the first.
                references.setdefault(
                    tuple(path.split(".")),
                    {
                        "path": uploaded_path,
                        "fileId": f"{{{{steps.{upload_id}.body.fileId}}}}",
                    },
                )
        if decl.webhook is not None:
            webhook_result = _webhook_result(artifact.output_schema, references)
    if decl.webhook is not None:
        result["steps"].append(
            {
                "id": WEBHOOK_STEP_ID,
                "name": decl.name + settings.MOBILERUN_SANDBOX_WEBHOOK_SUFFIX,
                "uses": "webhooks.send",
                "with": {
                    "endpointId": webhook_endpoint_id(decl.webhook),
                    "data": {"result": webhook_result},
                },
            }
        )
    return result


class _WebhookBindingError(SandboxDeploymentError):
    """Value-safe explanation of why the endpoint cannot be used."""


async def _check_webhook(
    endpoint_id: str | None, receipt: OwnershipReceipt, mode: Billing
) -> None:
    if endpoint_id is None:
        return
    endpoint_id = webhook_endpoint_id(endpoint_id)
    try:
        endpoint = await CloudWebhooks(billing=mode).get(endpoint_id)
    except CloudResourceNotFound:
        raise _WebhookBindingError(
            f"Webhook endpoint {endpoint_id} not found. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        ) from None
    if endpoint.state != "ACTIVE" or endpoint.health == "blocked":
        raise _WebhookBindingError(
            f"Webhook endpoint is disabled, deleted or blocked. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        )
    if endpoint.event_types and DEFAULT_WEBHOOK_EVENT not in endpoint.event_types:
        raise _WebhookBindingError(
            f"Webhook endpoint filter excludes {DEFAULT_WEBHOOK_EVENT}. Register one: {WEBHOOK_REGISTER_HINT}",
            receipt=receipt,
        )


def _action_type(action: JSON) -> str | None:
    uses = action.get("uses")
    if action.get("service") == "webhooks" and action.get("method") == "send":
        return "webhooks.send" if uses in (None, "", "webhooks.send") else None
    return uses if isinstance(uses, str) and "." in uses else None


def _same_json(left: Any, right: Any) -> bool:
    """Compare JSON intent without conflating booleans and numbers."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_same_json(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(_same_json(a, b) for a, b in zip(left, right, strict=True))
        )
    if type(left) in (int, float):
        return type(right) in (int, float) and left == right
    return type(left) is type(right) and left == right


def equivalent_document(actual: JSON, desired: JSON) -> bool:
    """Compare authored fields while detecting removed device/schema settings."""

    def compare(a: Any, d: Any) -> bool:
        if isinstance(d, dict):
            if not isinstance(a, dict):
                return False
            if "on" in d and "devices" not in d and a.get("devices"):
                return False
            if (
                "recording" in d
                and "delivery" not in d
                and a.get("delivery") is not None
            ):
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
                    _same_json(a[k], v)
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
        return _same_json(a, d)

    return compare(actual, desired)


@requires(Billing.PAID, operation="Sandbox deployment, endpoint run, or activation")
def _require_paid(*, billing: Billing) -> None:
    """Check commerce authorization before creating local or remote resources."""


async def _check_deploy_billing(billing: Billing) -> None:
    """Let interactive callers retry authorization without retrying inspection or deploy."""
    _require_paid(billing=billing)


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


async def _can_replace_initial_version(
    state: JSON,
    receipt: OwnershipReceipt,
    found: JSON | None,
    flows: list[JSON],
    mode: Billing,
) -> bool:
    """Only a rejected, receipt-owned version with no staged flow can be superseded."""
    if flows or receipt.flow_id or receipt.trigger_id or receipt.action_ids:
        return False
    if (
        state["function_pending"]
        or state["flow_pending"]
        or state["document"] is not None
    ):
        return False
    if not receipt.function_id or not found or found.get("id") != receipt.function_id:
        return False
    version = state["version"]
    if version is None or (found.get("latestVersion") or {}).get("version") != version:
        return False
    record = await clients.get_version(
        receipt.function_id, _version(version), billing=mode
    )
    return record.get("version") == version and record.get("status") == "rejected"


async def deploy_file(
    path: Path,
    *,
    device: str | None = None,
    device_free: bool = False,
    billing: Billing | str | None = None,
    artifact: Artifact | None = None,
) -> DeploymentReceipt:
    """Deploy a new function and optionally stage a first owned flow."""
    validate_device(device)
    mode = resolve_billing(billing)
    _require_paid(billing=mode)
    if artifact is None:
        artifact = package(path)
    elif artifact.declaration.schedule is not None:
        verify_scheduled_artifact(artifact)
    decl = artifact.declaration
    if device is not None and device_free:
        raise SandboxDeclarationError("Choose a device or device-free, not both")
    selected = None if device_free else device if device is not None else decl.device
    if decl.schedule is not None and selected is None and not device_free:
        raise SandboxDeclarationError(
            "Select a device or explicitly choose device-free"
        )
    # Validate declaration before any cloud calls or local writes.
    preview_document = build_document(
        artifact,
        "placeholder",
        1,
        device=selected,
    )
    if decl.has_endpoint and _has_ref(artifact.input_schema):
        raise SandboxDeclarationError("Endpoint schema contains unsupported $ref")
    scope = current_scope()
    store = ReceiptStore(scope, decl.name)
    with store.lock():
        receipt = store.read()
        initial = _initial(receipt) if receipt else None
        if initial and initial.get("creation_resolved"):
            raise SandboxDeploymentConflict(
                "Finish sandbox cleanup before deploying", receipt=receipt
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
        if initial and receipt is not None:
            if "preview_document" in initial:
                matches_document = _same_json(
                    initial["preview_document"], preview_document
                )
            elif (
                initial.get("document") is not None
                and receipt.function_id
                and initial.get("version")
            ):
                matches_document = _same_json(
                    initial["document"],
                    build_document(
                        artifact,
                        receipt.function_id,
                        initial["version"],
                        device=selected,
                    ),
                )
            else:
                if (
                    initial["function_pending"]
                    and receipt.function_id is None
                    and not initial["flow_pending"]
                    and not (
                        receipt.flow_id or receipt.trigger_id or receipt.action_ids
                    )
                ):
                    if initial["digest"] != artifact.digest:
                        raise SandboxDeploymentConflict(
                            "Retry with the original sandbox source to recover the function ID before cleanup",
                            receipt=receipt,
                        )
                    receipt = await _create_initial_function(
                        store, receipt, artifact, initial, found, mode
                    )
                raise SandboxDeploymentConflict(
                    "Initial intent evidence is missing; cleanup is required before redeploying",
                    receipt=receipt,
                )
            if (
                initial["digest"] != artifact.digest
                or initial["device"] != selected
                or not matches_document
            ):
                if initial[
                    "digest"
                ] == artifact.digest or not await _can_replace_initial_version(
                    initial, receipt, found, flows, mode
                ):
                    raise SandboxDeploymentConflict(
                        "Finish the original initial deployment first", receipt=receipt
                    )
                # No flow staging began. Replace only this confirmed terminal version.
                initial.update(
                    digest=artifact.digest,
                    device=selected,
                    preview_document=preview_document,
                    version=None,
                    replacing_rejected=True,
                )
                receipt = _checkpoint(store, receipt, initial)
            if initial.get("activation_pending"):
                receipt, flow = await _owned_remote(store, mode, activating=False)
                actual = await _remote_request(
                    clients.get_document(flow["id"], billing=mode)
                )
                function_id = _identifier(receipt.function_id)
                desired = build_document(
                    artifact,
                    function_id,
                    initial["version"],
                    device=selected,
                )
                if (
                    desired is None
                    or not _same_json(desired, initial["document"])
                    or not equivalent_document(actual, desired)
                ):
                    raise SandboxDeploymentConflict(
                        "Initial staging intent changed", receipt=receipt
                    )
                await _wait_ready(function_id, initial["version"], mode, receipt)
                return await _finish_initial_activation(
                    store, receipt, artifact, initial, mode, flow.get("enabled")
                )
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


async def _create_initial_function(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    state: JSON,
    found: JSON | None,
    mode: Billing,
) -> OwnershipReceipt:
    """Create or replay the journaled function only; never infer its ID from a name."""
    key = state["function_key"]
    receipt = store.record_stage(receipt, "function:create", key)
    try:
        result = await clients.create_function(
            receipt.name, artifact.source, key, billing=mode
        )
    except Exception:
        raise SandboxDeploymentError(
            "Sandbox function creation outcome uncertain; retry the original deployment",
            receipt=receipt,
        ) from None
    function_id = _identifier(result.get("id"))
    if receipt.function_id and receipt.function_id != function_id:
        raise SandboxDeploymentConflict(
            "Idempotent function response conflicts with remote state", receipt=receipt
        )
    receipt = store.record_ids(receipt, function_id=function_id)
    version = _version((result.get("latestVersion") or {}).get("version"))
    state.update(function_pending=False, version=version)
    receipt = _checkpoint(store, receipt, state)
    if found and found.get("id") != function_id:
        raise SandboxDeploymentConflict(
            "Idempotent function response conflicts with remote state", receipt=receipt
        )
    return receipt


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
    resuming = state is not None
    if state is None:
        await _check_webhook(decl.webhook, receipt, mode)
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
            preview_document=build_document(
                artifact,
                "placeholder",
                1,
                device=selected,
            ),
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
        receipt = await _create_initial_function(
            store, receipt, artifact, state, found, mode
        )
    elif state["version"] is None:
        if found is None:
            raise SandboxDeploymentConflict(
                "Owned sandbox function is missing", receipt=receipt
            )
        function_id = _identifier(found.get("id"))
        version = _version((found.get("latestVersion") or {}).get("version"))
        source = None
        if not state.get("replacing_rejected"):
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
    await _wait_ready(function_id, version, mode, receipt)
    receipt = store.record_ids(receipt, digest=artifact.digest)
    if state["flow_pending"]:
        # Replay the exact staged payload before checking external dependencies.
        document = state["document"]
    else:
        document = build_document(artifact, function_id, version, device=selected)
        if (
            document is not None
            and (await clients.validate_document(document, billing=mode)).get("valid")
            is not True
        ):
            raise SandboxDeploymentError(
                "Sandbox flow validation failed", receipt=receipt
            )
    if document is not None:
        key = state["flow_key"]
        if not state["flow_pending"]:
            if resuming:
                await _check_webhook(decl.webhook, receipt, mode)
            payloads = await clients.prepare_action_payloads(document, billing=mode)
            state.update(flow_pending=True, document=document, action_payloads=payloads)
            receipt = _checkpoint(store, receipt, state)
        elif not isinstance(state.get("action_payloads"), list):
            raise SandboxDeploymentConflict(
                "Cannot safely replay initial staging without journaled action payloads",
                receipt=receipt,
            )
        receipt = store.record_stage(receipt, "flow:create", key)
        try:
            staged = await clients.create_document(
                document, key, billing=mode, action_payloads=state["action_payloads"]
            )
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
        if len(set(action_ids)) != len(document["steps"]) or len(action_ids) != len(
            document["steps"]
        ):
            raise SandboxDeploymentError(
                "Incomplete sandbox action IDs", receipt=receipt
            )
        if any(f.get("id") != flow_id for f in visible_flows):
            raise SandboxDeploymentConflict(
                "Idempotent flow response conflicts with remote state", receipt=receipt
            )
        store.write_yaml(document)
        state.update(flow_pending=False, activation_pending=True)
        receipt = _checkpoint(store, receipt, state)
        return await _finish_initial_activation(
            store, receipt, artifact, state, mode, None
        )
    state["complete"] = True
    receipt = _checkpoint(store, receipt, state)
    return _deployment(receipt, artifact, version, state["status"])


async def _finish_initial_activation(
    store: ReceiptStore,
    receipt: OwnershipReceipt,
    artifact: Artifact,
    state: JSON,
    mode: Billing,
    current: bool | None,
) -> DeploymentReceipt:
    """Close staging only after the flow reflects the journaled activation intent."""
    flow_id = _identifier(receipt.flow_id)
    target = state.get("enabled", True)
    if target:
        await _check_activation_webhook(state["document"], receipt, mode)
    try:
        if current is not target:
            stage = "flow:enable" if target else "flow:disable"
            key = stage_key(store.scope, receipt.name, artifact.digest, stage)
            receipt = store.record_stage(receipt, stage, key)
            operation = clients.enable_flow if target else clients.disable_flow
            updated = await operation(flow_id, billing=mode)
            if not isinstance(updated, dict) or updated.get("enabled") is not target:
                raise ValueError("Activation not confirmed")
        if (await _flow(flow_id, mode))["enabled"] is not target:
            raise ValueError("Activation not confirmed")
    except Exception:
        raise SandboxDeploymentError(
            "Sandbox flow activation failed or uncertain", receipt=receipt
        ) from None
    state["complete"] = True
    receipt = _checkpoint(store, receipt, state)
    return _deployment(receipt, artifact, state["version"], state["status"])


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


def _require_current_cutover_receipt(
    state: JSON | None, receipt: OwnershipReceipt
) -> None:
    if state is not None and "action_payloads" not in state:
        raise SandboxDeploymentConflict(
            "Sandbox cutover receipt predates this version; finish or clean up the "
            "deployment manually. See 'Local store and ownership' in the sandbox README.",
            receipt=receipt,
        )


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


def _without_webhook(document: JSON) -> JSON:
    intent = copy.deepcopy(document)
    intent["steps"] = [
        step for step in intent["steps"] if step["uses"] != "webhooks.send"
    ]
    # Adding/removing the sink can rename the function for legacy templates.
    intent["steps"][0].pop("name")
    return intent


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
    state = _cutover(receipt)
    _require_current_cutover_receipt(state, receipt)
    if not found or not receipt.trigger_id or not receipt.action_ids:
        raise SandboxDeploymentConflict(
            "Sandbox ownership is incomplete", receipt=receipt
        )
    _identifier(receipt.function_id)
    flow_id = _identifier(receipt.flow_id)
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
        requested = build_document(
            artifact, receipt.function_id, state["version"], device=selected
        )
        desired_changed = not _same_json(requested, state["desired"])
        retarget = (
            state.get("webhook_unusable")
            and desired_changed
            and requested is not None
            and _same_json(
                _without_webhook(requested), _without_webhook(state["desired"])
            )
        )
        if (
            state["digest"] != artifact.digest
            or state["device"] != selected
            or (desired_changed and not retarget)
        ):
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
        delivery_candidates = []
        if "delivery" in state["done"]:
            delivery_candidates = candidates
        elif state.get("pending") == "delivery":
            delivery_candidates = [copy.deepcopy(candidates[0])]
            candidates.extend(delivery_candidates)
        for candidate in delivery_candidates:
            settings = candidate.setdefault("settings", {})
            if delivery := state["desired"].get("settings", {}).get("delivery"):
                settings["delivery"] = delivery
            else:
                settings.pop("delivery", None)
        if not any(equivalent_document(actual, c) for c in candidates):
            raise SandboxDeploymentConflict(
                "Sandbox remote state diverged", receipt=receipt
            )
        version = state["version"]
        await _wait_ready(receipt.function_id, version, mode, receipt)
        desired = state["desired"]
        if retarget and requested is not None:
            await _check_activation_webhook(requested, receipt, mode)
            if (await clients.validate_document(requested, billing=mode)).get(
                "valid"
            ) is not True:
                raise SandboxDeploymentError(
                    "Sandbox flow validation failed", receipt=receipt
                )
            # The observed document is a verified cutover state, not new ownership.
            # Keep every known action until the replacement is confirmed active.
            state.update(
                before=actual,
                desired=requested,
                action_payloads=await clients.prepare_action_payloads(
                    requested, billing=mode
                ),
                old_ids=list(bindings),
                retire_ids=list(receipt.action_ids),
                new_ids=[],
                done=[],
                pending=None,
                webhook_unusable=False,
                retarget_generation=len(receipt.stages),
            )
            receipt = _checkpoint(store, receipt, state)
            desired = requested
    else:
        has_yaml = store.yaml_path.exists()
        authored = _read_authored_document(store, receipt)
        if has_yaml or authored is not None:
            if (
                not isinstance(authored, dict)
                or (not has_yaml and not _has_authored_steps(authored))
                or not equivalent_document(actual, authored)
            ):
                raise SandboxDeploymentConflict(
                    "Sandbox remote document changed", receipt=receipt
                )
        pinned = _pinned_version(receipt.function_id, actual, for_redeploy=True)
        version = pinned
        source = await clients.get_source(receipt.function_id, version, billing=mode)
        if source != artifact.source:
            version = _version((found.get("latestVersion") or {}).get("version"))
            if version != pinned:
                source = await clients.get_source(
                    receipt.function_id, version, billing=mode
                )
        if source == artifact.source:
            await _wait_ready(receipt.function_id, version, mode, receipt)
        await _check_webhook(artifact.declaration.webhook, receipt, mode)
        desired = build_document(
            artifact,
            receipt.function_id,
            version,
            device=selected,
        )
        if desired is None:
            raise SandboxDeploymentConflict(
                "Remove the owned flow explicitly first", receipt=receipt
            )
        if source == artifact.source and equivalent_document(actual, desired):
            if artifact.declaration.webhook and not store.yaml_path.exists():
                store.write_yaml(desired)
            return _deployment(receipt, artifact, version, "unchanged")
        if authored is None:
            raise SandboxDeploymentConflict(
                "Sandbox authored document evidence is missing", receipt=receipt
            )
        if source != artifact.source:
            key = stage_key(
                store.scope,
                receipt.name,
                artifact.digest,
                f"version:create:{receipt.function_id}:{pinned}:{','.join(bindings)}",
            )
            receipt = store.record_stage(receipt, "version:create", key)
            result = await clients.create_version(
                receipt.function_id, artifact.source, key, billing=mode
            )
            version = _version(result.get("version"))
            await _wait_ready(receipt.function_id, version, mode, receipt)
            desired = build_document(
                artifact,
                receipt.function_id,
                version,
                device=selected,
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
            action_payloads=await clients.prepare_action_payloads(
                desired, billing=mode
            ),
            old_ids=list(receipt.action_ids),
            new_ids=[],
            enabled=flow["enabled"],
            done=[],
            pending=None,
            complete=False,
        )
        receipt = _checkpoint(store, receipt, state)

    payloads = state.get("action_payloads")
    if not isinstance(payloads, list) or len(payloads) != len(desired["steps"]):
        raise SandboxDeploymentConflict(
            "Cannot safely resume cutover without journaled action payloads",
            receipt=receipt,
        )

    def save():
        nonlocal receipt
        receipt = _checkpoint(store, receipt, state)

    def intent(stage):
        nonlocal receipt
        state["pending"] = stage
        save()
        # Includes the full desired document, so device/schema-only changes get new keys.
        identity = [state["desired"], state["old_ids"]]
        if "retarget_generation" in state:
            identity.append(state["retarget_generation"])
        digest = _hash(json.dumps(identity, sort_keys=True))
        key = stage_key(store.scope, receipt.name, digest, stage)
        receipt = store.record_stage(receipt, stage, key)
        return key

    def done(stage):
        state["done"].append(stage)
        state["pending"] = None
        save()

    delivery = desired.get("settings", {}).get("delivery")
    delivery_changed = not _same_json(
        state["before"].get("settings", {}).get("delivery"), delivery
    )

    async def update_delivery() -> None:
        if delivery_changed and "delivery" not in state["done"]:
            key = intent("delivery")
            await clients.update_flow_delivery(
                _identifier(receipt.flow_id), delivery, key, billing=mode
            )
            done("delivery")

    try:
        activated = "enable" in state["done"]
        if not activated:
            # Re-prove shutdown, including after an uncertain enable. Once enable
            # is confirmed, retries only verify the live flow and retire old actions.
            key = stage_key(store.scope, receipt.name, artifact.digest, "disable")
            receipt = store.record_stage(receipt, "disable", key)
            await clients.disable_flow(receipt.flow_id, billing=mode)
            if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown not confirmed")
            if "disable" not in state["done"]:
                state["done"].append("disable")
                save()
        for index, payload in enumerate(payloads):
            stage = f"create:{index}"
            if stage in state["done"]:
                continue
            key = intent(stage)
            action = await clients.create_action(
                payload,
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
        # The service permits uploads without delivery, but not the reverse.
        if delivery is None:
            await update_delivery()
        if "replace" not in state["done"]:
            if (await _flow(receipt.flow_id, mode))["enabled"] is not False:
                raise ValueError("Shutdown no longer confirmed")
            await _check_activation_webhook(desired, receipt, mode)
            key = intent("replace")
            refs = [
                dict(action_id=a, position=i, key=desired["steps"][i]["id"])
                for i, a in enumerate(state["new_ids"])
            ]
            await clients.replace_flow_actions(receipt.flow_id, refs, key, billing=mode)
            done("replace")
        if (
            not _same_json(state["before"]["on"], desired["on"])
            and "trigger" not in state["done"]
        ):
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
            not _same_json(
                state["before"].get("devices", []), desired.get("devices", [])
            )
            and "devices" not in state["done"]
        ):
            key = intent("devices")
            await clients.update_flow_devices(
                receipt.flow_id, desired.get("devices", []), key, billing=mode
            )
            done("devices")
        if delivery is not None:
            await update_delivery()
        if not activated:
            intent("verify")
        await _unshared(receipt, mode)
        expected_enabled = state["enabled"] if activated else False
        if (await _flow(flow_id, mode))[
            "enabled"
        ] is not expected_enabled or not equivalent_document(
            await clients.get_document(flow_id, billing=mode), desired
        ):
            raise ValueError("Cutover verification failed")
        verified = tuple(
            a.get("actionId") for a in await clients.list_actions(flow_id, billing=mode)
        )
        if verified != tuple(state["new_ids"]):
            raise ValueError("Cutover bindings changed")
        if not activated:
            done("verify")
        if state["enabled"] and not activated:
            await _check_activation_webhook(desired, receipt, mode)
            intent("enable")
            await clients.enable_flow(flow_id, billing=mode)
            if (await _flow(flow_id, mode))["enabled"] is not True:
                raise ValueError("Enable not confirmed")
            done("enable")
        unresolved = []
        retire_ids: list[str] = (
            state["retire_ids"] if "retire_ids" in state else state["old_ids"]
        )
        for action_id in retire_ids:
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
        state["complete"] = True
        save()
        warnings = (
            () if state["enabled"] else ("Flow remains disabled (enabled=False)",)
        )
        return _deployment(receipt, artifact, version, "pushed", warnings)
    except Exception as exc:
        # A confirmed activation must not be undone by a cleanup failure.
        try:
            restore_old = (
                isinstance(exc, _WebhookBindingError)
                and state["enabled"]
                and not {"replace", "delivery"}.intersection(state["done"])
                and state.get("pending") not in ("replace", "delivery")
            )
            if restore_old:
                # Only definitions were created; no live binding or settings changed.
                await _unshared(receipt, mode)
                current = await clients.get_document(flow_id, billing=mode)
                current_ids = [
                    row.get("actionId")
                    for row in await clients.list_actions(flow_id, billing=mode)
                ]
                if current_ids != state["old_ids"] or not equivalent_document(
                    current, state["before"]
                ):
                    raise ValueError("Previous workflow state changed")
                await _check_activation_webhook(state["before"], receipt, mode)
                key = stage_key(
                    store.scope, receipt.name, artifact.digest, "restore:enable"
                )
                receipt = store.record_stage(receipt, "restore:enable", key)
                await clients.enable_flow(flow_id, billing=mode)
            elif "enable" not in state["done"]:
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
        if isinstance(exc, _WebhookBindingError):
            state["webhook_unusable"] = True
            state[
                "warning"
            ] += f". {exc}. Redeploy with a usable endpoint ID or remove webhook=."
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
            or receipt.origin != store.scope.origin
            or receipt.account_id != store.scope.account_id
        ):
            raise SandboxDeploymentConflict(
                "Sandbox cleanup receipt mismatch", receipt=stored
            )
        state = _cutover(receipt)
        _require_current_cutover_receipt(state, receipt)
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
        if state and str(state.get("pending", "")).startswith("create:"):
            # Preserve the flow and receipt so deploy can replay the original create.
            raise SandboxDeploymentError(
                "Sandbox action creation outcome uncertain; retry the original deploy before cleanup",
                receipt=receipt,
            )
        if initial:
            # Preserve resolved creation evidence before cleanup clears any IDs.
            initial["creation_resolved"] = True
            receipt = _checkpoint(store, receipt, initial)

        def intent(stage: str, resource_id: str) -> None:
            nonlocal receipt
            key = stage_key(store.scope, receipt.name, resource_id, "cleanup:" + stage)
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
                if not flow_deleted or receipt.action_ids:
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
        if state:
            state["complete"] = True
            _checkpoint(store, receipt, state)


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
) -> tuple[OwnershipReceipt, JSON]:
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
    return receipt, flow


def _read_authored_document(store: ReceiptStore, receipt: OwnershipReceipt) -> Any:
    """Read local YAML or the most recent completed deployment's document."""
    if store.yaml_path.exists():
        try:
            return yaml.safe_load(store.yaml_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, yaml.YAMLError):
            raise SandboxDeploymentConflict(
                "Sandbox document evidence is unreadable", receipt=receipt
            ) from None
    for entry in reversed(receipt.stages):
        state = json.loads(entry)
        if state.get("complete") and state.get("stage") in ("initial", "cutover"):
            key = "document" if state["stage"] == "initial" else "desired"
            return state.get(key)
    return None


def _has_authored_steps(document: Any) -> bool:
    # Equivalence allows extra remote fields, so require authored action bodies.
    steps = document.get("steps") if isinstance(document, dict) else None
    return (
        isinstance(steps, list)
        and bool(steps)
        and all(
            isinstance(step, dict) and {"id", "name", "uses", "with"} <= step.keys()
            for step in steps
        )
    )


async def _activation_document(
    store: ReceiptStore, receipt: OwnershipReceipt, flow: JSON, mode: Billing
) -> JSON:
    document = await _remote_request(clients.get_document(flow["id"], billing=mode))
    live_steps = document.get("steps")
    has_webhook = isinstance(live_steps, list) and any(
        _action_type(step) == "webhooks.send"
        for step in live_steps
        if isinstance(step, dict)
    )
    has_yaml = store.yaml_path.exists()
    authored = _read_authored_document(store, receipt)
    if not has_yaml and authored is None and not has_webhook:
        return document
    if has_webhook and not _has_authored_steps(authored):
        raise SandboxDeploymentConflict(
            "Missing authored webhook flow evidence; redeploy before activation",
            receipt=receipt,
        )
    if not isinstance(authored, dict) or not equivalent_document(document, authored):
        raise SandboxDeploymentConflict(
            "Sandbox remote document changed; redeploy before activation",
            receipt=receipt,
        )
    return document


def _pinned_version(
    function_id: str, document: JSON, *, for_redeploy: bool = False
) -> int:
    def is_webhook(step: Any) -> bool:
        return (
            isinstance(step, dict)
            and step.get("uses") == "webhooks.send"
            and step.get("id") == WEBHOOK_STEP_ID
        )

    steps = document.get("steps", [])
    optional_steps = steps[1:] if isinstance(steps, list) else []
    if optional_steps and is_webhook(optional_steps[-1]):
        optional_steps = optional_steps[:-1]
    elif optional_steps and is_webhook(optional_steps[0]):
        if not for_redeploy:
            raise SandboxDeploymentConflict(
                "Sandbox webhook precedes file uploads; redeploy before activation"
            )
        optional_steps = optional_steps[1:]
    if (
        not isinstance(steps, list)
        or not steps
        or not isinstance(steps[0], dict)
        or steps[0].get("uses") != "functions.run"
        or any(
            not isinstance(step, dict)
            or step.get("uses") != FILE_UPLOAD_USES
            or step.get("id") != f"upload_{index}"
            for index, step in enumerate(optional_steps, 1)
        )
    ):
        raise SandboxDeploymentConflict(
            "Sandbox flow must contain its owned function step"
        )
    params = steps[0].get("with")
    if not isinstance(params, dict) or params.get("functionId") != function_id:
        raise SandboxDeploymentConflict("Sandbox flow function binding changed")
    return _version(params.get("version"))


async def _flow_version(function_id: str, document: JSON, mode: Billing) -> JSON:
    version = _pinned_version(function_id, document)
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
        receipt, flow = await _owned_remote(store, mode, activating=True)
        document = await _activation_document(store, receipt, flow, mode)
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


async def _check_activation_webhook(
    document: JSON, receipt: OwnershipReceipt, mode: Billing
) -> None:
    for step in document["steps"]:
        if step.get("uses") == "webhooks.send":
            await _check_webhook(
                _identifier((step.get("with") or {}).get("endpointId")), receipt, mode
            )


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
        receipt, flow = await _owned_remote(store, mode, activating=enabled)
        if enabled:
            document = await _activation_document(store, receipt, flow, mode)
            await _flow_version(receipt.function_id, document, mode)
            await _check_activation_webhook(document, receipt, mode)
        key = stage_key(
            store.scope,
            name,
            receipt.digest or "",
            "flow:enable" if enabled else "flow:disable",
        )
        if not enabled:
            pending = _cutover(receipt) or _initial(receipt)
            if pending:
                pending["enabled"] = False
                receipt = _checkpoint(store, receipt, pending)
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
    for key in ("flowId", "invocationId", "startedAt", "finishedAt"):
        if isinstance(record.get(key), str):
            result[key] = record[key]
    # Use this execution's own progress, not the current (possibly redeployed) flow.
    for source in ("progress", "result"):
        metadata = record.get(source)
        if not isinstance(metadata, dict) or not isinstance(
            metadata.get("steps"), list
        ):
            continue
        if source == "result":
            result["result"] = {"steps": []}
        for step in metadata["steps"]:
            if not isinstance(step, dict):
                continue
            uses = _action_type(step)
            status = step.get("status")
            body = step.get("body")
            if source == "result" and uses == "functions.run":
                function: JSON = {"uses": uses}
                if status is not None:
                    function["status"] = status if status == "success" else "failed"
                if isinstance(body, dict):
                    function["body"] = {
                        k: copy.deepcopy(body[k])
                        for k in (
                            "result",
                            "outcome",
                            "ok",
                            "partialReason",
                            "state",
                            "notes",
                        )
                        if k in body
                    }
                    if body.get("error") is not None:
                        function["status"] = "failed"
                result["result"]["steps"].append(function)
            if uses != "webhooks.send":
                continue
            if isinstance(body, dict) and body.get("error") is not None:
                status = "failed"
            elif status is None and isinstance(body, dict) and body.get("eventId"):
                status = "succeeded"
            if status == "success":
                status = "succeeded"
            if status in ("succeeded", "failed", "skipped", "cancelled"):
                result["webhookStatus"] = status
            elif status in ("pending", "running") and result.get("webhookStatus") in (
                None,
                "pending",
            ):
                result["webhookStatus"] = status
            if source == "result":
                safe_step: JSON = {"uses": "webhooks.send"}
                if "webhookStatus" in result:
                    safe_step["status"] = result["webhookStatus"]
                if isinstance(body, dict) and isinstance(body.get("eventId"), str):
                    safe_step["body"] = {"eventId": body["eventId"]}
                result["result"]["steps"].append(safe_step)
    return result


async def execution_snapshot(
    execution_id: str, *, billing: Billing | str | None = None
) -> JSON:
    """Retrieve a fresh execution snapshot, preserving partial result metadata."""
    return _snapshot(await _execution_record(execution_id, billing))


async def _execution_record(execution_id: str, billing: Billing | str | None) -> JSON:
    execution_id = _run_identifier(execution_id)
    record = await _remote_request(
        clients.get_execution(execution_id, billing=resolve_billing(billing))
    )
    if record.get("id") != execution_id:
        raise SandboxRunError(
            "Sandbox execution ID mismatch", execution_id=execution_id
        )
    return record


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
        if isinstance(s, dict) and _action_type(s) == "functions.run"
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
        webhook = snapshot.get("webhookStatus")
        webhook_failed = webhook in ("failed", "skipped", "cancelled")
        if status == "success" or (
            status in ("failed", "cancelled") and webhook_failed
        ):
            _check_success(snapshot, invocation_id)
            if webhook_failed:
                logging.getLogger(__name__).warning(
                    "Sandbox webhook step %s; function result is available", webhook
                )
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
    record = await _execution_record(execution_id, billing)
    for step in _function_steps(record):
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
