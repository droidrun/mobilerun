"""Flow document construction and comparison."""

import copy
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .errors import SandboxDeclarationError
from .models import (
    FILE_MIME_TYPES,
    FILE_UPLOAD_USES,
    JSON,
    Artifact,
    is_result_path_part,
    webhook_endpoint_id,
)
from .settings import WEBHOOK_STEP_ID, get_sandbox_settings


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


def action_type(action: JSON) -> str | None:
    """Return the ``service.method`` of a remote action, or None when it is unclear."""
    uses = action.get("uses")
    if action.get("service") == "webhooks" and action.get("method") == "send":
        return "webhooks.send" if uses in (None, "", "webhooks.send") else None
    return uses if isinstance(uses, str) and "." in uses else None


def same_json(left: Any, right: Any) -> bool:
    """Compare JSON intent without conflating booleans and numbers."""
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(same_json(value, right[key]) for key, value in left.items())
        )
    if isinstance(left, list):
        return (
            isinstance(right, list)
            and len(left) == len(right)
            and all(same_json(a, b) for a, b in zip(left, right, strict=True))
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
                    same_json(a[k], v)
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
        return same_json(a, d)

    return compare(actual, desired)


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


def has_ref(value: Any) -> bool:
    """Return whether a JSON schema contains a reference at any depth."""
    if not isinstance(value, dict):
        return False
    return any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")) or any(
        has_ref(child) for child in _schema_children(value)
    )


def without_webhook(document: JSON) -> JSON:
    """Return a copy of the document without its webhook step and function step name."""
    intent = copy.deepcopy(document)
    intent["steps"] = [
        step for step in intent["steps"] if step["uses"] != "webhooks.send"
    ]
    # Adding/removing the sink can rename the function for legacy templates.
    intent["steps"][0].pop("name")
    return intent


def endpoint_schema(document: JSON) -> JSON | None:
    """Return the custom-trigger input schema of a flow document, if it has one."""
    custom = document.get("on", {}).get("custom")
    if isinstance(custom, dict) and isinstance(custom.get("inputs"), dict):
        return custom["inputs"]
    return None


def has_authored_steps(document: Any) -> bool:
    """Return whether every step carries the fields a locally authored document has."""
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
