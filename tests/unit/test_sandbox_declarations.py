import asyncio
import importlib
import runpy

import pytest
from mobilerun_functions.entry import take

import mobilerun.sandbox as sandbox
from mobilerun.sandbox.errors import SandboxDeclarationError


@pytest.mark.parametrize("is_async", [False, True])
def test_no_deploy_on_import_or_main(tmp_path, monkeypatch, is_async):
    def tripwire(*args, **kwargs):
        pytest.fail("declaration attempted deployment, transport, or inspection")

    operations = importlib.import_module("mobilerun.sandbox.deployment")
    for name in ("deploy_file", "describe_remote", "fire_remote", "package"):
        monkeypatch.setattr(operations, name, tripwire)
    monkeypatch.setattr("socket.socket.connect", tripwire)
    monkeypatch.setattr("mobilerun.sandbox.packaging.inspect_source", tripwire)
    path = tmp_path / "declared.py"
    path.write_text("""\
from __future__ import annotations
from pydantic import BaseModel
from mobilerun.sandbox import cloud

class Input(BaseModel):
    n: int

class Output(BaseModel):
    n: int

def main(value: Input) -> Output | None:
    return Output(n=value.n + 1)

original = main
main = cloud('demo')(main)
""".replace("def main", "async def main" if is_async else "def main"))
    for module in ("declared", "__main__"):
        try:
            namespace = runpy.run_path(str(path), run_name=module)
            main = namespace["main"]
            assert main is namespace["original"]
            output = main(namespace["Input"](n=3))
            assert (asyncio.run(output) if is_async else output).n == 4
            assert isinstance(main.remote, sandbox.Remote)
            assert main.remote.input_type is namespace["Input"]
            assert main.remote.output_type == namespace["Output"] | None
            assert take(module).func is main
        finally:
            take(module)


@pytest.mark.parametrize(
    "options",
    [
        dict(name=" "),
        dict(name=42),
        dict(schedule="* * * * * *"),
        dict(schedule=42),
        dict(device=42),
        dict(device=""),
        dict(device=" \t"),
        dict(endpoint=1),
        dict(schedule="* * * * *", endpoint=True),
    ],
)
def test_invalid_cloud_options(options):
    with pytest.raises(SandboxDeclarationError):
        sandbox.cloud(**{"name": "demo", **options})


def test_unknown_cloud_option_is_rejected():
    with pytest.raises(TypeError):
        sandbox.cloud("demo", unknown=True)


def test_cloud_declaration_rejects_blank_device():
    from mobilerun.sandbox.models import CloudDeclaration

    with pytest.raises(SandboxDeclarationError, match="Device"):
        CloudDeclaration("demo", device=" ").validate()


def test_manual_trigger_is_not_a_decorator_parameter():
    with pytest.raises(TypeError, match="trigger"):
        sandbox.cloud("demo", trigger="manual")


def test_unresolved_cloud_signature_is_rejected():
    def main(input: "MissingModel") -> None:  # noqa: F821
        pass

    with pytest.raises(SandboxDeclarationError, match="annotations"):
        sandbox.cloud("demo")(main)
    assert take(__name__) is None
