"""Sandbox declarations, source extraction, and artifact inspection."""

import asyncio
import importlib
import runpy
import traceback
from pathlib import Path
from textwrap import dedent

import pytest
from mobilerun_functions.entry import take

import mobilerun.sandbox as sandbox
from mobilerun.sandbox.errors import SandboxDeclarationError, SandboxPackagingError
from mobilerun.sandbox.packaging import package, read_declaration, transform


def test_malformed_decorator_traceback_omits_source(tmp_path):
    path = tmp_path / "invalid.py"
    token = "traceback-token-sentinel"
    path.write_text(
        f'from mobilerun.sandbox import cloud\n@cloud(webhook="https://example.test/?token={token}"))\n'
    )
    with pytest.raises(SandboxDeclarationError) as caught:
        package(path)
    formatted = "".join(traceback.format_exception(caught.value))
    assert token not in formatted
    assert f"{path}:2" in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None


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
        dict(webhook="http://localhost/callback"),
        dict(webhook="https:///missing-host"),
        dict(webhook="https://user:password@example.com"),
        dict(webhook="https://example.com", endpoint=False),
        dict(schedule="* * * * *", endpoint=True),
    ],
)
def test_invalid_cloud_options(options):
    with pytest.raises(SandboxDeclarationError):
        sandbox.cloud(**{"name": "demo", **options})


def test_unknown_cloud_option_is_rejected():
    with pytest.raises(TypeError):
        sandbox.cloud("demo", unknown=True)


def test_static_declaration_rejects_receiver_url_without_echo(tmp_path):
    path = tmp_path / "routine.py"
    path.write_text(
        'from mobilerun.sandbox import cloud\n@cloud("demo", webhook="https://example.test/?token=sentinel")\ndef demo(): pass\n'
    )
    with pytest.raises(SandboxDeclarationError) as caught:
        read_declaration(path)
    assert (
        str(caught.value)
        == f"{path}:2: webhook= takes a webhook endpoint ID. Register one first: mobilerun webhook register <url>"
    )
    assert "sentinel" not in "".join(traceback.format_exception(caught.value))


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


SOURCE = """from mobilerun.sandbox import cloud, Context
from pydantic import BaseModel
from datetime import datetime, timezone

class Item(BaseModel):
    at: datetime

class Input(BaseModel):
    items: list[Item]

UNUSED = 1

@cloud("example")
def main(input: Input, ctx: Context) -> Input | None:
    return input
"""


def test_extracts_nested_models_and_required_imports_only():
    result = transform(SOURCE)
    assert "class Item(" in result
    assert "class Input(" in result
    assert "from datetime import datetime" in result
    assert "timezone" not in result
    assert "UNUSED" not in result
    assert "from mobilerun_functions import " in result
    assert result.count("@script") == 1


def test_rejects_external_global_with_source_line():
    source = SOURCE.replace("return input", "return UNUSED")
    with pytest.raises(SandboxPackagingError, match=r"main.py:15:.*UNUSED"):
        transform(source)


def test_rejects_multiple_cloud_functions():
    source = SOURCE + '\n@cloud("other")\ndef other(): pass\n'
    with pytest.raises(SandboxDeclarationError):
        transform(source)


@pytest.mark.parametrize(
    "expression",
    [
        "os.path",
        "helper()",
        "UNUSED",
        "[json.dumps(i) for i in range(3)]",
        "(lambda: helper())()",
        "(lambda: UNUSED)()",
    ],
)
def test_global_reads_in_nested_scopes_are_rejected(expression):
    with pytest.raises(SandboxPackagingError, match="main.py:15:"):
        transform(SOURCE.replace("return input", "return " + expression))


def test_lazy_import_nested_closure_and_comprehension():
    source = SOURCE.replace(
        "    return input",
        '    import os\n    def inner(x):\n        return os.path.join(str(x), "a")\n    values = [inner(i) for i in range(3)]\n    return input',
    )
    assert "    import os" in transform(source)


def test_comprehension_shadow_does_not_hide_outer_global():
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(
            SOURCE.replace("return input", "return [UNUSED for UNUSED in []], UNUSED")
        )


def test_comprehension_target_shadows_global():
    transform(SOURCE.replace("return input", "return [UNUSED for UNUSED in []]"))


def test_global_statement_rejected():
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(
            SOURCE.replace("    return input", "    global UNUSED\n    return UNUSED")
        )


def test_all_global_read_lines_reported():
    source = SOURCE.replace(
        "    return input", "    print(UNUSED)\n    return helper()"
    )
    with pytest.raises(SandboxPackagingError) as error:
        transform(source)
    assert "main.py:15: 'UNUSED'" in str(error.value)
    assert "main.py:16: 'helper'" in str(error.value)


def test_shadow_reports_only_actual_global_read():
    source = SOURCE.replace(
        "    return input", "    values = [UNUSED for UNUSED in []]\n    return UNUSED"
    )
    with pytest.raises(SandboxPackagingError) as error:
        transform(source)
    assert str(error.value).count("'UNUSED'") == 1
    assert "main.py:16:" in str(error.value)


def test_nested_function_annotations_checked():
    source = SOURCE.replace(
        "    return input",
        "    def inner(x: UNUSED):\n        return x\n    return input",
    )
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(source)


def test_future_annotations_forward_models():
    source = (
        SOURCE.replace(
            "from mobilerun.sandbox",
            "from __future__ import annotations\nfrom mobilerun.sandbox",
        )
        .replace("class Item(BaseModel):\n    at: datetime\n\n", "")
        .replace("UNUSED = 1", "class Item(BaseModel):\n    at: datetime\n\nUNUSED = 1")
        .replace("-> Input | None", '-> "Input | None"')
    )
    result = transform(source)
    assert "from __future__ import annotations" in result
    assert "class Item(" in result
    assert "class Input(" in result


def test_multiple_lambdas_on_same_line():
    with pytest.raises(SandboxPackagingError) as error:
        transform(
            SOURCE.replace("return input", "return (lambda: helper(), lambda: UNUSED)")
        )
    assert "'helper'" in str(error.value)
    assert "'UNUSED'" in str(error.value)


def test_dataset_nested_model():
    result = transform(
        SOURCE.replace("cloud, Context", "cloud, Context, Dataset").replace(
            "-> Input | None", "-> Dataset[Item]"
        )
    )
    assert "Dataset" in result
    assert "class Item(" in result


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace(
            "from mobilerun.sandbox import cloud, Context",
            "from mobilerun.sandbox import Context",
        ),
        SOURCE.replace('@cloud("example")', "@script"),
        SOURCE.replace(
            "from mobilerun.sandbox import cloud, Context",
            "from mobilerun.sandbox import cloud\nfrom mobilerun.sandbox import cloud, Context",
        ),
        SOURCE.replace('@cloud("example")', '@cloud("example")\n@cloud("other")'),
        SOURCE.replace('@cloud("example")', '@other\n@cloud("example")'),
        SOURCE.replace("import cloud, Context", "import cloud as c, Context"),
    ],
)
def test_invalid_declarations_rejected(source):
    with pytest.raises(SandboxDeclarationError):
        transform(source)


@pytest.mark.parametrize(
    "change",
    [
        ("from datetime import datetime, timezone", "from .sibling import datetime"),
        ("    return input", "    from .sibling import thing\n    return thing"),
        (
            "def main(input: Input, ctx: Context)",
            "def main(input: Input, ctx: Context = UNUSED)",
        ),
        ("    at: datetime", "    at: datetime = UNUSED"),
    ],
)
def test_unsupported_dependencies_rejected(change):
    with pytest.raises(SandboxPackagingError):
        transform(SOURCE.replace(*change))


def test_external_constant_in_annotation_is_rejected():
    with pytest.raises(SandboxPackagingError, match="UNUSED"):
        transform(SOURCE.replace("input: Input", "input: UNUSED"))


def test_model_method_local_variables_are_not_global_dependencies():
    source = SOURCE.replace(
        "    at: datetime",
        "    at: datetime\n    def display(self):\n        value = str(self.at)\n        return value",
    )
    assert "def display" in transform(source)


def test_model_method_global_helper_is_rejected():
    source = SOURCE.replace(
        "    at: datetime",
        "    at: datetime\n    def display(self):\n        return helper(self.at)",
    )
    with pytest.raises(SandboxPackagingError, match="helper"):
        transform(source)


def test_sibling_lazy_import_rejected(tmp_path):
    (tmp_path / "sibling.py").write_text("value = 1\n")
    source = SOURCE.replace(
        "    return input", "    import sibling\n    return sibling.value"
    )
    with pytest.raises(SandboxPackagingError, match="local import"):
        transform(source, filename=str(tmp_path / "main.py"))


def test_nested_quoted_model_reference_is_extracted():
    source = SOURCE.replace("items: list[Item]", 'items: list["Item"]')
    assert "class Item(" in transform(source)


def test_shadowed_builtin_in_model_is_rejected():
    source = SOURCE.replace("UNUSED = 1", "str = helper()\nUNUSED = 1")
    source = source.replace("    at: datetime", "    at: str")
    with pytest.raises(SandboxPackagingError, match="'str'"):
        transform(source)


def test_model_quoted_external_reference_is_rejected():
    source = SOURCE.replace("    at: datetime", '    at: "MissingType"')
    with pytest.raises(SandboxPackagingError, match="MissingType"):
        transform(source)


def test_literal_string_values_are_not_forward_references():
    source = SOURCE.replace(
        "from pydantic import BaseModel",
        "from pydantic import BaseModel\nfrom typing import Literal",
    )
    source = source.replace("    at: datetime", '    at: Literal["ready", "waiting"]')
    result = transform(source)
    assert 'Literal["ready", "waiting"]' in result


def test_separate_runtime_alias_is_preserved():
    source = SOURCE.replace(
        "from pydantic import BaseModel",
        "from pydantic import BaseModel\nfrom mobilerun_functions import Dataset as Data",
    )
    source = source.replace("-> Input | None", "-> Data[Item]")
    assert "Dataset as Data" in transform(source)


def test_model_method_local_does_not_authorize_entry_global():
    source = """from mobilerun.sandbox import cloud
from pydantic import BaseModel
MISSING = 42
class Input(BaseModel):
    def display(self):
        MISSING = 1
        return MISSING
@cloud("example")
def main(input: Input):
    print(MISSING)
"""
    with pytest.raises(SandboxPackagingError, match="MISSING"):
        transform(source)


@pytest.mark.parametrize(
    "imports,annotation",
    [
        ("from typing import Literal as L", 'L["ready", "waiting"]'),
        ("import typing", 'typing.Literal["ready", "waiting"]'),
        ("from typing import Annotated as A", 'A[str, "metadata"]'),
        ("import typing as t", 't.Annotated[str, "metadata"]'),
    ],
)
def test_annotation_containers_resolve_import_bindings(imports, annotation):
    source = SOURCE.replace(
        "from pydantic import BaseModel", "from pydantic import BaseModel\n" + imports
    )
    source = source.replace("    at: datetime", "    at: " + annotation)
    assert annotation in transform(source)


@pytest.mark.parametrize(
    "prefix,decorator",
    [
        ("", "@cloud"),
        ("alias = cloud\n", '@alias("other")'),
        ("import mobilerun.sandbox as sandbox\n", '@sandbox.cloud("other")'),
        (
            "import mobilerun.sandbox as sandbox\nalias = sandbox.cloud\n",
            '@alias("other")',
        ),
    ],
)
def test_additional_unsupported_cloud_declaration_rejected(prefix, decorator):
    source = SOURCE + "\n" + prefix + decorator + "\ndef other(): pass\n"
    with pytest.raises(SandboxDeclarationError):
        transform(source)


def test_model_method_local_annotation_does_not_authorize_global():
    source = """from mobilerun.sandbox import cloud
from pydantic import BaseModel
MISSING = 42
class Input(BaseModel):
    def display(self):
        local: MISSING = 1
        return local
@cloud("example")
def main(input: Input):
    print(MISSING)
"""
    with pytest.raises(SandboxPackagingError, match="MISSING"):
        transform(source)
