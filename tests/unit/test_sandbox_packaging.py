from pathlib import Path

import pytest

from mobilerun.sandbox.packaging import package, read_declaration


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

    def fail(_source):
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
