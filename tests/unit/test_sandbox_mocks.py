import asyncio
import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import mobilerun.sandbox as sandbox
from mobilerun.models.billing import Billing


def test_explicit_deploy_is_exported_even_after_loading_submodule(monkeypatch):
    import mobilerun.sandbox.deployment as operations

    receipt = object()
    fake = AsyncMock(return_value=receipt)
    monkeypatch.setattr(operations, "deploy_file", fake)
    monkeypatch.setenv("MOBILERUN_BILLING", "no_commerce")

    def confirm(preview):
        return True

    assert sandbox.deploy("main.py", device_free=True, confirm=confirm) is receipt
    fake.assert_awaited_once_with(
        Path("main.py"),
        device=None,
        device_free=True,
        billing=Billing.NO_COMMERCE,
        yes=False,
        confirm=confirm,
    )
    assert sandbox.deploy("main.py", billing="paid", yes=True) is receipt
    assert fake.await_args.kwargs["billing"] == Billing.PAID


@pytest.mark.parametrize(
    "name", ["App", "Task", "SandboxNotImplemented", "DeployedScript", "RunResult"]
)
def test_obsolete_public_names_are_gone(name):
    assert not hasattr(sandbox, name)


def test_obsolete_client_is_gone():
    assert importlib.util.find_spec("mobilerun.sandbox.client") is None


def test_obsolete_sandbox_error_class_is_gone():
    assert not hasattr(
        importlib.import_module("mobilerun.sandbox.errors"), "SandboxNotImplemented"
    )


def test_python_deploy_rejects_blank_device_before_packaging(monkeypatch):
    from mobilerun.sandbox import deployment
    from mobilerun.sandbox.errors import SandboxDeclarationError

    def tripwire(path):
        pytest.fail("invalid device reached packaging")

    monkeypatch.setattr(deployment, "package", tripwire)
    with pytest.raises(SandboxDeclarationError, match="Device"):
        sandbox.deploy("main.py", device=" \t", yes=True)


def test_deploy_file_previews_supplied_artifact_without_rereading(
    tmp_path, monkeypatch
):
    from mobilerun.sandbox import deployment
    from mobilerun.sandbox.errors import SandboxConfirmationRequired
    from mobilerun.sandbox.models import Artifact, CloudDeclaration

    path = tmp_path / "changed.py"
    path.write_text("not the confirmed source")
    artifact = Artifact(CloudDeclaration("confirmed"), "", "digest", {}, None)
    monkeypatch.setattr(
        deployment, "package", lambda _: pytest.fail("unexpected repackaging")
    )

    def deny(preview):
        assert preview.name == "confirmed"
        return False

    with pytest.raises(SandboxConfirmationRequired):
        asyncio.run(deployment.deploy_file(path, confirm=deny, artifact=artifact))
