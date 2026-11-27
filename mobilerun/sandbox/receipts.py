"""Local ownership receipts and the deployment journal."""

import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import yaml

from mobilerun.clients.local.login_store import resolve_cloud_api_key
from mobilerun.config.settings import get_settings
from mobilerun.utils.paths import local_store_path

from .errors import SandboxDeploymentConflict, SandboxDeploymentError
from .models import JSON, OwnershipReceipt
from .settings import SandboxSettings, get_sandbox_settings


def _windows_platform() -> bool:
    return sys.platform == "win32"


@dataclass(frozen=True)
class Scope:
    """Cloud origin and API-key fingerprint; never the API key itself."""

    origin: str
    account_id: str


def sha256_hex(value: str) -> str:
    """Return the SHA-256 hex digest of a string."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def current_scope() -> Scope:
    """Resolve credentials exactly as the shared cloud client does."""
    key = resolve_cloud_api_key()
    if not key:
        raise SandboxDeploymentError("Sandbox credentials required")
    base = httpx.URL(str(get_settings().MOBILERUN_CLOUD_BASE_URL))
    path = base.raw_path
    base = base.copy_with(raw_path=path if path.endswith(b"/") else path + b"/")
    return Scope(str(base), sha256_hex(key))


def stage_key(scope: Scope, name: str, digest: str, stage: str) -> str:
    """Return a stable key for the same scoped mutation intent."""
    return sha256_hex(
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
            root / sha256_hex(scope.origin + "\0" + scope.account_id) / sha256_hex(name)
        )
        # Keep a legacy receipt, its authored YAML, lock and retry keys together.
        if scope.origin.endswith("/") and not scope.origin.endswith("//"):
            legacy = scope.origin[:-1]
            prior = (
                root / sha256_hex(legacy + "\0" + scope.account_id) / sha256_hex(name)
            )
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


def pending_initial(receipt: OwnershipReceipt) -> JSON | None:
    """Return the unfinished first-deploy stage recorded in a receipt, if any."""
    for entry in reversed(receipt.stages):
        value = json.loads(entry)
        if value.get("stage") == "initial":
            return value if not value.get("complete") else None
    return None


def pending_cutover(receipt: OwnershipReceipt) -> JSON | None:
    """Return the unfinished redeploy cutover stage recorded in a receipt, if any."""
    for entry in reversed(receipt.stages):
        value = json.loads(entry)
        if value.get("stage") == "cutover":
            return value if not value.get("complete") else None
    return None


def require_current_cutover_receipt(
    state: JSON | None, receipt: OwnershipReceipt
) -> None:
    """Reject a cutover stage journaled without action payloads by an older version."""
    if state is not None and "action_payloads" not in state:
        raise SandboxDeploymentConflict(
            "Sandbox cutover receipt predates this version; finish or clean up the "
            "deployment manually. See 'Local store and ownership' in the sandbox README.",
            receipt=receipt,
        )


def checkpoint(
    store: ReceiptStore, receipt: OwnershipReceipt, state: JSON
) -> OwnershipReceipt:
    """Append a stage to the receipt journal and write it to the store."""
    result = replace(
        receipt, stages=receipt.stages + (json.dumps(state, sort_keys=True),)
    )
    store.write(result)
    return result


def read_authored_document(store: ReceiptStore, receipt: OwnershipReceipt) -> Any:
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
