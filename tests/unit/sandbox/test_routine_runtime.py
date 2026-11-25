"""Sandbox runtime selection, routine execution, and public contracts."""

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import mobilerun_functions
import mobilerun_functions.context
import pytest
from mobilerun_functions.cli import main


def test_billing_precedence(monkeypatch):
    from mobilerun.models.billing import Billing
    from mobilerun.sandbox.models import resolve_billing

    monkeypatch.delenv("MOBILERUN_BILLING", raising=False)
    assert resolve_billing(None) == Billing.PAID
    monkeypatch.setenv("MOBILERUN_BILLING", "included")
    assert resolve_billing(None) == Billing.INCLUDED
    assert resolve_billing("no_commerce") == Billing.NO_COMMERCE


def test_receipt_scope_excludes_credentials(monkeypatch):
    import hashlib
    import os

    secret = "sentinel-secret"
    monkeypatch.setenv("MOBILERUN_API_KEY", secret)
    fingerprint = hashlib.sha256(secret.encode()).hexdigest()
    from mobilerun.sandbox.models import DeploymentReceipt, OwnershipReceipt

    assert os.environ["MOBILERUN_API_KEY"] == secret

    receipt = OwnershipReceipt(
        "https://example.test", fingerprint, "name", None, None, None, (), None, ()
    )
    assert dataclasses.asdict(receipt)["origin"] == "https://example.test"
    assert dataclasses.asdict(receipt)["account_id"] == fingerprint
    assert receipt.to_dict() == {
        "origin": "https://example.test",
        "account_id": fingerprint,
        "name": "name",
        "function_id": None,
        "flow_id": None,
        "trigger_id": None,
        "action_ids": [],
        "digest": None,
        "stages": [],
    }
    assert secret not in str(receipt.to_dict())
    deployment = DeploymentReceipt(
        "name", "function", 1, "created", None, None, (), "digest", None, ()
    )
    assert secret not in str(deployment.to_dict())
    with pytest.raises(dataclasses.FrozenInstanceError):
        receipt.name = "other"


def test_sandbox_directory_default_override_empty(tmp_path, monkeypatch):
    from mobilerun.sandbox import settings
    from mobilerun.utils.paths import local_store_path

    monkeypatch.delenv("MOBILERUN_SANDBOX_DIRECTORY", raising=False)
    monkeypatch.setattr(
        settings,
        "get_settings",
        lambda: SimpleNamespace(MOBILERUN_USER_CONFIG_DIR=tmp_path / "missing"),
    )
    assert settings.SandboxSettings().sandbox_directory == str(
        tmp_path / "missing" / "sandbox"
    )
    assert not (tmp_path / "missing").exists()
    for override in ("relative/path", "~/sandbox-test"):
        config = settings.SandboxSettings(MOBILERUN_SANDBOX_DIRECTORY=override)
        assert config.sandbox_directory == override
        assert local_store_path(override, overridden=True).path == str(
            Path(override).expanduser().absolute()
        )
    assert (
        settings.SandboxSettings(
            MOBILERUN_SANDBOX_DIRECTORY=""
        ).MOBILERUN_SANDBOX_DIRECTORY
        is None
    )


def test_contract_dataclasses_and_errors():
    from mobilerun.models.billing import Billing
    from mobilerun.sandbox import errors, models

    declaration = models.CloudDeclaration("hello")
    artifact = models.Artifact(declaration, "source", "digest", {}, None)
    assert artifact.declaration == declaration
    assert (
        models.DeploymentPreview("hello", None, None, Billing.PAID).billing
        == Billing.PAID
    )
    endpoint = models.Endpoint("https://example.test", "trigger")
    receipt = models.DeploymentReceipt(
        "hello", "f", 1, "created", None, None, (), "digest", endpoint, ()
    )
    assert receipt.endpoint == endpoint
    assert receipt.to_dict() == {
        "name": "hello",
        "function_id": "f",
        "version": 1,
        "status": "created",
        "flow_id": None,
        "trigger_id": None,
        "action_ids": [],
        "digest": "digest",
        "endpoint": {"url": "https://example.test", "trigger_id": "trigger"},
        "warnings": [],
    }
    ownership = models.OwnershipReceipt(
        "origin", "account", "name", None, None, None, (), None, ()
    )
    for cls in (
        errors.SandboxDeclarationError,
        errors.SandboxPackagingError,
        errors.SandboxConfirmationRequired,
    ):
        assert isinstance(cls("safe"), errors.SandboxError)
    for cls in (errors.SandboxDeploymentError, errors.SandboxDeploymentConflict):
        assert cls("safe", receipt=ownership).receipt == ownership
    for cls in (errors.SandboxRunError, errors.SandboxRunTimeout):
        error = cls("safe", invocation_id="i", execution_id="e")
        assert (error.invocation_id, error.execution_id) == ("i", "e")
    assert isinstance(errors.SandboxRunTimeout("timeout"), TimeoutError)


@pytest.mark.parametrize("missing", ["mobilerun_functions", "nested_dependency"])
def test_runtime_import_reports_missing_dependency(missing):
    import subprocess
    import sys

    code = """\
import importlib.abc
import sys
missing = sys.argv[1]
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobilerun_functions":
            raise ModuleNotFoundError("blocked", name=missing)
sys.meta_path.insert(0, BlockRuntime())
import mobilerun.sandbox
try:
    mobilerun.sandbox.Context
except ImportError as exc:
    if missing == "mobilerun_functions":
        assert "mobilerun[sandbox]" in str(exc)
    else:
        assert isinstance(exc, ModuleNotFoundError) and exc.name == missing
else:
    raise AssertionError("expected missing dependency failure")
"""
    subprocess.run([sys.executable, "-c", code, missing], check=True)


def test_runtime_selection(monkeypatch):
    from importlib import import_module

    from mobilerun.sandbox import _runtime
    from mobilerun.sandbox.settings import RUNTIME_MODULE, SandboxSettings

    monkeypatch.setenv("runtime_module", "wrong_runtime")
    assert RUNTIME_MODULE == "mobilerun_functions"
    assert "runtime_module" not in SandboxSettings.model_fields
    assert _runtime.script is import_module(RUNTIME_MODULE).script


@pytest.mark.parametrize(
    ("signature", "body", "expected", "kind", "is_async", "warning"),
    [
        (
            "def main(input: Input) -> Output:",
            "return Output(value=input.value)",
            {"value": "hello"},
            "value",
            False,
            False,
        ),
        (
            "async def main(input: Input) -> Output:",
            "return Output(value=input.value)",
            {"value": "hello"},
            "value",
            True,
            False,
        ),
        (
            "def main(input: Input) -> Output | None:",
            "return None",
            None,
            "value",
            False,
            False,
        ),
        ("def main(input: Input):", "return None", None, "none", False, True),
        (
            "def main(input: Input, ctx: Context) -> Output:",
            "return Output(value=ctx.run.run_id)",
            {"value": "local"},
            "value",
            False,
            False,
        ),
        (
            "def main(input: Input) -> Dataset[Item]:",
            "return Dataset(items=[Item(id='one', nested=Output(value=input.value))])",
            {"items": [{"id": "one", "nested": {"value": "hello"}}]},
            "dataset",
            False,
            False,
        ),
    ],
)
def test_selected_runtime_inspect_run_parity(
    tmp_path, signature, body, expected, kind, is_async, warning
):
    import json
    import subprocess
    import sys

    from mobilerun.sandbox.settings import RUNTIME_MODULE

    input_file = tmp_path / "input.json"
    input_file.write_text('{"value":"hello"}')
    documents = []
    envelopes = []
    for module, decorator in (
        ("mobilerun.sandbox", "routine"),
        (RUNTIME_MODULE, "script"),
    ):
        source = tmp_path / f"{decorator}.py"
        source.write_text(f"""\
from pydantic import BaseModel
from {module} import {decorator}, Context, Dataset, key_field
class Input(BaseModel):
    value: str
class Output(BaseModel):
    value: str
class Item(BaseModel):
    id: str = key_field()
    nested: Output
@{decorator}
""" + signature + "\n    " + body + "\n")
        result_file = tmp_path / f"{decorator}.json"
        inspect = subprocess.run(
            [sys.executable, "-m", RUNTIME_MODULE, "inspect", str(source)],
            capture_output=True,
            text=True,
            check=True,
        )
        documents.append(json.loads(inspect.stdout))
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                RUNTIME_MODULE,
                "run",
                str(source),
                "--input-file",
                str(input_file),
                "--result-file",
                str(result_file),
            ],
            capture_output=True,
            text=True,
        )
        assert run.returncode == 0, run.stderr
        envelopes.append(json.loads(result_file.read_text()))
    assert documents[0] == documents[1]
    assert envelopes[0] == envelopes[1]
    document = documents[0]
    assert document["ok"] is True
    assert document["async"] is is_async
    assert document["outputKind"] == kind
    assert document["inputSchema"]["properties"]["value"]["type"] == "string"
    assert bool(document["warnings"]) is warning
    if kind != "none":
        assert document["outputSchema"] is not None
    assert envelopes[0] == {
        "ok": True,
        "outcome": "succeeded",
        "result": expected,
        "input": {"value": "hello"},
    }


def test_settings_defaults_and_error_optional_ids():
    from mobilerun.sandbox.errors import SandboxDeploymentError, SandboxRunError
    from mobilerun.sandbox.settings import get_sandbox_settings

    settings = get_sandbox_settings()
    assert (
        settings.MOBILERUN_SANDBOX_INSPECTION_TIMEOUT,
        settings.MOBILERUN_SANDBOX_POLL_INTERVAL,
        settings.MOBILERUN_SANDBOX_RUN_TIMEOUT,
    ) == (120.0, 2.0, 300.0)
    assert (
        settings.MOBILERUN_SANDBOX_FUNCTION_TIMEOUT,
        settings.MOBILERUN_SANDBOX_FLOW_SUFFIX,
        settings.MOBILERUN_SANDBOX_ACTION_SUFFIX,
    ) == (600, " · flow", " · function")
    assert SandboxDeploymentError("stage").receipt is None
    error = SandboxRunError("stage")
    assert error.invocation_id is None and error.execution_id is None


def test_routine_returns_original_callable():
    import subprocess
    import sys

    source = """\
from pydantic import BaseModel
from mobilerun.sandbox import routine
class Input(BaseModel): pass
def main(input: Input) -> None: pass
assert routine(main) is main
"""
    subprocess.run([sys.executable, "-c", source], check=True)


def test_sandbox_tunable_environment_names(monkeypatch):
    from mobilerun.sandbox.settings import SandboxSettings

    monkeypatch.setenv("MOBILERUN_SANDBOX_POLL_INTERVAL", "4.5")
    assert SandboxSettings().MOBILERUN_SANDBOX_POLL_INTERVAL == 4.5


def test_public_routine_api_is_reexported():
    import mobilerun.sandbox as sandbox

    for name in mobilerun_functions.__all__:
        assert getattr(sandbox, name) is getattr(mobilerun_functions, name)
    assert callable(sandbox.routine)


@pytest.mark.parametrize("async_prefix", ["", "async "])
def test_routine_matches_script_inspect_and_run(
    tmp_path, monkeypatch, capsys, async_prefix
):
    template = """\
from pydantic import BaseModel
from {module} import {decorator}, Context

class Input(BaseModel):
    message: str

class Output(BaseModel):
    value: str

@{decorator}
{async_prefix}def main(input: Input, ctx: Context) -> Output:
    return Output(value=ctx.device().echo(input.message))
"""
    paths = []
    for module, decorator in (
        ("mobilerun.sandbox", "routine"),
        ("mobilerun_functions", "script"),
    ):
        path = tmp_path / f"{decorator}.py"
        path.write_text(
            template.format(
                module=module, decorator=decorator, async_prefix=async_prefix
            )
        )
        paths.append(path)

    inspected = []
    for path in paths:
        assert main(["inspect", str(path)]) == 0
        inspected.append(json.loads(capsys.readouterr().out))
    assert inspected[0] == inspected[1]
    assert inspected[0]["ok"] is True

    class FakeDevice:
        def echo(self, message):
            return f"device: {message}"

    monkeypatch.setattr(
        mobilerun_functions.context,
        "connect_device",
        lambda device_id, **kwargs: FakeDevice(),
    )
    input_file = tmp_path / "input.json"
    input_file.write_text('{"message":"hello"}')
    envelopes = []
    for path in paths:
        result_file = tmp_path / f"{path.stem}.json"
        assert (
            main(
                [
                    "run",
                    str(path),
                    "--input-file",
                    str(input_file),
                    "--result-file",
                    str(result_file),
                    "--device-id",
                    "fake",
                ]
            )
            == 0
        )
        envelopes.append(json.loads(result_file.read_text()))
    assert (
        envelopes
        == [
            {
                "ok": True,
                "outcome": "succeeded",
                "result": {"value": "device: hello"},
                "input": {"message": "hello"},
            }
        ]
        * 2
    )
