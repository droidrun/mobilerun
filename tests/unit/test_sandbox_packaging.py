from pathlib import Path
from textwrap import dedent

import pytest

from mobilerun.sandbox.packaging import package, read_declaration


@pytest.mark.anyio
@pytest.mark.parametrize("supplied", ["path", "inspected", "raw", "edited"])
async def test_before_validator_can_fill_required_schedule_field(
    tmp_path, supplied, monkeypatch
):
    from dataclasses import replace
    from unittest.mock import Mock

    from mobilerun.sandbox import deployment, packaging
    from mobilerun.sandbox.errors import SandboxConfirmationRequired
    from mobilerun.sandbox.models import Artifact

    path = tmp_path / "task.py"
    path.write_text(dedent("""\
        from pydantic import BaseModel, model_validator
        from mobilerun.sandbox import cloud
        class Input(BaseModel):
            count: int
            @model_validator(mode='before')
            @classmethod
            def fill_default(cls, data):
                return {'count': 1, **data}
        @cloud('demo', schedule='* * * * *')
        def task(input: Input) -> Input: return input
    """))
    inspected = package(path)
    assert inspected.input_schema["required"] == ["count"]
    raw = Artifact(
        inspected.declaration,
        inspected.source,
        inspected.digest,
        inspected.input_schema,
        inspected.output_schema,
    )
    inspect = Mock(wraps=packaging.inspect_source)
    monkeypatch.setattr(packaging, "inspect_source", inspect)
    artifact = {
        "path": None,
        "inspected": inspected,
        "raw": raw,
        "edited": replace(inspected, source=inspected.source + "\n"),
    }[supplied]
    with pytest.raises(SandboxConfirmationRequired):
        await deployment.deploy_file(path, device_free=True, artifact=artifact)
    assert inspect.call_count == (0 if supplied == "inspected" else 1)


def test_schedule_inspects_real_model_before_cloud(tmp_path, monkeypatch):
    import asyncio

    from jsonschema import validate

    from mobilerun.sandbox import deployment
    from mobilerun.sandbox.errors import SandboxDeclarationError

    monkeypatch.setattr(
        deployment, "current_scope", lambda: pytest.fail("cloud access")
    )
    source = tmp_path / "main.py"
    template = dedent("""\
        from pydantic import BaseModel, Field, field_validator
        from mobilerun.sandbox import cloud

        raise AssertionError('original module executed')

        class Input(BaseModel):
            count: int = Field(default=12, validate_default=True)

            @field_validator('count')
            @classmethod
            def reject_default(cls, value):
                raise ValueError('private-validator-detail')

        @cloud('demo'OPTIONS)
        def task(input: Input) -> Input:
            raise AssertionError('entry point ran')
    """)
    source.write_text(template.replace("OPTIONS", ""))
    validate({}, package(source).input_schema)
    source.write_text(template.replace("OPTIONS", ", schedule='* * * * *'"))
    with pytest.raises(SandboxDeclarationError, match="count") as caught:
        asyncio.run(deployment.deploy_file(source, device_free=True, yes=True))
    assert "private-validator-detail" not in str(caught.value)
    assert "12" not in str(caught.value)
    source.write_text(
        source.read_text().replace(
            "raise ValueError('private-validator-detail')", "return value"
        )
    )
    assert package(source).declaration.schedule == "* * * * *"


def test_package_does_not_execute_original_module(tmp_path: Path):
    sentinel = tmp_path / "sentinel"
    source = tmp_path / "main.py"
    source.write_text(
        "from mobilerun.sandbox import cloud\n"
        "from pydantic import BaseModel\n"
        "class Input(BaseModel):\n    value: int\n"
        f'open({str(sentinel)!r}, "w").write("bad")\n'
        '@cloud("example")\n'
        "def main(value: Input) -> Input:\n    return value\n"
    )
    assert read_declaration(source).name == "example"
    artifact = package(source)
    assert artifact.declaration.name == "example"
    assert not sentinel.exists()


@pytest.mark.parametrize(
    "decorator", ["@cloud(name=external)", '@cloud("x", endpoint=external)']
)
def test_nonliteral_declaration_rejected(tmp_path: Path, decorator: str):
    from mobilerun.sandbox.errors import SandboxDeclarationError

    source = tmp_path / "main.py"
    source.write_text(
        f"from mobilerun.sandbox import cloud\n{decorator}\ndef main(): pass\n"
    )
    with pytest.raises(SandboxDeclarationError):
        read_declaration(source)


def test_inspection_failure_stops_package(tmp_path, monkeypatch):
    from mobilerun.sandbox import clients as functions
    from mobilerun.sandbox import packaging
    from mobilerun.sandbox.errors import SandboxPackagingError

    source = tmp_path / "main.py"
    source.write_text(
        'from mobilerun.sandbox import cloud\n@cloud("x")\ndef main(): pass\n'
    )
    calls = []

    def boundary(*args, **kwargs):
        calls.append("cloud")
        raise AssertionError("cloud boundary called")

    monkeypatch.setattr(functions, "cloud_client", boundary)

    def fail(_source, *, scheduled):
        calls.append("inspect")
        raise SandboxPackagingError("inspection failed")

    monkeypatch.setattr(packaging, "inspect_source", fail)
    with pytest.raises(SandboxPackagingError):
        package(source)
    assert calls == ["inspect"]


def test_main_guard_not_executed(tmp_path):
    sentinel = tmp_path / "sentinel"
    source = tmp_path / "main.py"
    source.write_text(
        'from mobilerun.sandbox import cloud\nfrom pydantic import BaseModel\nclass Input(BaseModel):\n    value: int\n@cloud("x")\ndef main(input: Input) -> Input: return input\n'
        + f'if __name__ == "__main__":\n    open({str(sentinel)!r}, "w").write("bad")\n'
    )
    package(source)
    assert not sentinel.exists()


def test_inspection_uses_runtime_timeout_and_cleans_up(monkeypatch):
    import json
    import sys
    from types import SimpleNamespace

    from mobilerun.sandbox import packaging
    from mobilerun.sandbox.settings import RUNTIME_MODULE

    seen = []

    def run(command, **kwargs):
        path = Path(command[-1])
        assert path.read_text() == "trusted source"
        assert command[:-1] == [sys.executable, "-m", RUNTIME_MODULE, "inspect"]
        assert kwargs["timeout"] == 3
        seen.append(path)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"ok": True, "inputSchema": {}, "outputSchema": None}),
        )

    monkeypatch.setattr(
        packaging,
        "get_sandbox_settings",
        lambda: SimpleNamespace(MOBILERUN_SANDBOX_INSPECTION_TIMEOUT=3),
    )
    monkeypatch.setattr(packaging.subprocess, "run", run)
    assert packaging.inspect_source("trusted source")["ok"] is True
    assert not seen[0].exists()


@pytest.mark.parametrize(
    "stdout,returncode",
    [
        ("not json", 0),
        ("secret", 0),
        ('{"ok": false}', 0),
        ('{"ok": true}', 0),
        ("secret", 1),
    ],
)
def test_inspection_failures_are_sanitized(monkeypatch, stdout, returncode):
    import traceback
    from types import SimpleNamespace

    from mobilerun.sandbox import packaging
    from mobilerun.sandbox.errors import SandboxPackagingError

    monkeypatch.setattr(
        packaging.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=stdout, returncode=returncode),
    )
    with pytest.raises(SandboxPackagingError) as error:
        packaging.inspect_source("trusted source")
    assert "secret" not in str(error.value)
    assert "secret" not in "".join(traceback.format_exception(error.value))


def test_digest_includes_runtime_and_exact_generated_bytes(tmp_path):
    import hashlib

    from mobilerun.sandbox.settings import RUNTIME_MODULE

    source = tmp_path / "main.py"
    source.write_text(
        'from mobilerun.sandbox import cloud\nfrom pydantic import BaseModel\nclass Input(BaseModel):\n    value: int\n@cloud("x")\ndef main(input: Input) -> Input: return input\n'
    )
    artifact = package(source)
    assert (
        artifact.digest
        == hashlib.sha256(
            RUNTIME_MODULE.encode() + b"\0" + artifact.source.encode()
        ).hexdigest()
    )
    assert artifact.input_schema["type"] == "object"


def test_timeout_error_and_cleanup_are_sanitized(monkeypatch):
    import subprocess

    from mobilerun.sandbox import packaging
    from mobilerun.sandbox.errors import SandboxPackagingError

    paths = []

    def run(command, **kwargs):
        paths.append(Path(command[-1]))
        raise subprocess.TimeoutExpired(command, 1, output="secret")

    monkeypatch.setattr(packaging.subprocess, "run", run)
    with pytest.raises(SandboxPackagingError) as error:
        packaging.inspect_source("trusted source")
    assert "secret" not in str(error.value)
    assert not paths[0].exists()
