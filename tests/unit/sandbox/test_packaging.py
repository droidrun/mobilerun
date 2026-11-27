"""Sandbox declarations, source extraction, and artifact inspection."""

import asyncio
import runpy
import traceback
from pathlib import Path
from textwrap import dedent, indent

import pytest
from mobilerun_functions.entry import take

import mobilerun.sandbox as sandbox
from mobilerun.sandbox.errors import SandboxDeclarationError, SandboxPackagingError
from mobilerun.sandbox.packaging import package, read_declaration, transform
from tests.support.sandbox_fixtures import patch_sandbox


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

    for name in ("deploy_file", "describe_remote", "fire_remote", "package"):
        patch_sandbox(monkeypatch, name, tripwire)
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


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (dict(endpoint=False), "flow trigger"),
        (dict(files=[]), "files must be a dict"),
        (dict(files_folder=None), "safe folder"),
        (dict(files_folder="  "), "safe folder"),
        (dict(files_folder="one/../two"), "safe folder"),
        (dict(files_folder="/"), "safe folder"),
        (dict(files_folder="/one/./two"), "safe folder"),
        (dict(files_folder="one//two"), "safe folder"),
        (dict(files_folder="//one"), "safe folder"),
        (dict(files_folder="one/"), "safe folder"),
        (dict(files_folder="a" * 1024), "safe folder"),
        (dict(files_folder="😀" * 512), "safe folder"),
        (dict(files_folder="one\x7f"), "safe folder"),
        (dict(files={"": "image"}), "safe single path segment"),
        (dict(files={1: "image"}), "safe single path segment"),
        (dict(files={"a" * 252 + ".png": "image"}), "safe single path segment"),
        (dict(files={"one/two.jpg": "image"}), "safe single path segment"),
        (dict(files={"one\\two.jpg": "image"}), "safe single path segment"),
        (dict(files={"one\x00.jpg": "image"}), "safe single path segment"),
        (dict(files={" map.jpg": "image"}), "safe single path segment"),
        (dict(files={"map.jpg\u00a0": "image"}), "safe single path segment"),
        (dict(files={"😀" * 126 + ".jpg": "image"}), "safe single path segment"),
        (dict(files={"not.gif": "image"}), "unsupported file extension"),
        (dict(files={"A.jpg": "image", "a.JPG": "image"}), "filename collision"),
        *[
            (dict(files={"valid.jpg": path}), "invalid output field")
            for path in (
                1,
                "",
                "events..image",
                "events.01.image",
                "events.-1.image",
                "events[0].image",
                "events.__proto__.image",
                "events.Prototype.image",
                "events.constructor.image",
            )
        ],
    ],
)
def test_invalid_file_options(options, message):
    with pytest.raises(SandboxDeclarationError, match=message):
        sandbox.cloud(
            "demo",
            **({"files": {"valid.jpg": "image"}, "files_folder": "ok"} | options),
        )


def test_file_name_accepts_exactly_255_utf16_units():
    sandbox.cloud("demo", files={"😀" * 125 + "a.jpg": "image"}, files_folder="ok")


@pytest.mark.parametrize("annotation", ["", " -> None", " -> Output | None"])
def test_file_upload_requires_nonoptional_model_on_import_and_packaging(
    tmp_path, annotation
):
    path = tmp_path / "requires_model.py"
    path.write_text(dedent(f"""\
        from __future__ import annotations
        from pydantic import BaseModel
        from mobilerun.sandbox import cloud

        class Output(BaseModel):
            image: str

        @cloud('demo', files={{'image.jpg': 'image'}}, files_folder='images')
        def task(value: Output){annotation}:
            raise AssertionError('entry point ran')
    """))
    with pytest.raises(SandboxDeclarationError, match="non-optional.*BaseModel"):
        runpy.run_path(str(path), run_name="requires_model")
    assert take("requires_model") is None
    with pytest.raises(SandboxDeclarationError, match="non-optional.*BaseModel"):
        package(path)


@pytest.mark.parametrize(
    ("fields", "field_path", "message"),
    [
        ("events: list[Event] = Field(min_length=1)", "events.0.missing", "required"),
        ("event: Event | None", "event.image", "required, non-null"),
        (
            "events: list[Event] = Field(min_length=1, max_length=1)",
            "events.1.image",
            "min_length=2",
        ),
        ("image: str | None", "image", "non-null string"),
        ("image: int", "image", "non-null string"),
        ("image: str = ''", "image", "required"),
        ("events: list[Event]", "events.0.image", "min_length=1"),
        ("events: list[Event] = Field(min_length=1)", "events.1.image", "min_length=2"),
    ],
)
def test_packaging_rejects_file_paths_not_guaranteed_by_output_schema(
    tmp_path, fields, field_path, message
):
    path = tmp_path / "routine.py"
    path.write_text(dedent(f"""\
        from pydantic import BaseModel, Field
        from mobilerun.sandbox import cloud

        class Input(BaseModel):
            pass

        class Event(BaseModel):
            image: str

        class Output(BaseModel):
            {fields}

        @cloud('demo', files_folder='images', files={{'map.jpg': {field_path!r}}})
        def task(value: Input) -> Output:
            raise AssertionError('entry point ran')
    """))
    with pytest.raises(SandboxDeclarationError, match=message) as caught:
        package(path)
    assert field_path in str(caught.value)


def test_packaged_schedule_builds_file_uploads_and_webhook(tmp_path):
    from mobilerun.sandbox.documents import build_document

    path = tmp_path / "routine.py"
    path.write_text(dedent("""\
        from pydantic import BaseModel, Field
        from mobilerun.sandbox import cloud

        class Input(BaseModel):
            pass

        class Event(BaseModel):
            title: str | None = Field(default=None, serialization_alias='event-title')
            tags: list[str]
            map_data: str = Field(serialization_alias='map_image')
            transcript_image: str

        class Output(BaseModel):
            captured: str = Field(serialization_alias='captured_at')
            events: list[Event] = Field(min_length=3, max_length=3)

        @cloud('demo', schedule='*/30 * * * *', device='phone',
               files_folder='/atc',
               files={
                   'event_1_map.JPEG': 'events.0.map_image',
                   'event_1_transcript.png': 'events.0.transcript_image',
                   'event_2_map.jpg': 'events.1.map_image',
                   'event_2_transcript.jpg': 'events.1.transcript_image',
                   'event_3_map.jpg': 'events.2.map_image',
                   'event_3_transcript.jpg': 'events.2.transcript_image',
               },
               webhook='abcdef00-0000-4000-8000-000000000001')
        def task(value: Input) -> Output:
            raise AssertionError('entry point ran')
    """))
    try:
        assert runpy.run_path(str(path), run_name="file_schedule")["task"].remote
    finally:
        take("file_schedule")
    doc = build_document(package(path), "fn", 7, device="phone")
    assert doc["on"] == {"schedule": {"cron": "*/30 * * * *", "timezone": "UTC"}}
    assert doc["devices"] == ["phone"]
    assert doc["settings"] == {
        "recording": {"mode": "off"},
        "delivery": {"destination": "mobilerun", "folder": "/atc"},
    }
    function, *uploads, webhook = doc["steps"]
    jpg, png, *_ = uploads
    assert function["id"] == function["name"] == "function"
    assert function["with"]["version"] == 7
    assert [step["id"] for step in uploads] == [f"upload_{n}" for n in range(1, 7)]
    assert all(step["uses"] == "files.upload" for step in uploads)
    assert jpg == {
        "id": "upload_1",
        "name": "upload_1",
        "uses": "files.upload",
        "with": {
            "filename": "event_1_map.JPEG",
            "mimeType": "image/jpeg",
            "content": "{{steps.function.body.result.events.0.map_image}}",
            "contentEncoding": "base64",
        },
    }
    assert png == {
        "id": "upload_2",
        "name": "upload_2",
        "uses": "files.upload",
        "with": {
            "filename": "event_1_transcript.png",
            "mimeType": "image/png",
            "content": "{{steps.function.body.result.events.0.transcript_image}}",
            "contentEncoding": "base64",
        },
    }
    assert webhook["uses"] == "webhooks.send"
    assert webhook["with"] == {
        "endpointId": "abcdef00-0000-4000-8000-000000000001",
        "data": {
            "result": {
                "captured_at": "{{steps.function.body.result.captured_at}}",
                "events": [
                    {
                        "event-title": f"{{{{steps.function.body.result.events.{i}.event-title}}}}",
                        "tags": f"{{{{steps.function.body.result.events.{i}.tags}}}}",
                        "map_image": {
                            "path": f"/atc/event_{i + 1}_map.{extension}",
                            "fileId": f"{{{{steps.upload_{2 * i + 1}.body.fileId}}}}",
                        },
                        "transcript_image": {
                            "path": f"/atc/event_{i + 1}_transcript.{transcript}",
                            "fileId": f"{{{{steps.upload_{2 * i + 2}.body.fileId}}}}",
                        },
                    }
                    for i, (extension, transcript) in enumerate(
                        [("JPEG", "png"), ("jpg", "jpg"), ("jpg", "jpg")]
                    )
                ],
            }
        },
    }


@pytest.mark.parametrize(
    "exclude", ["exclude=True", "exclude_if=lambda value: not value"]
)
def test_webhook_rejects_omittable_serialized_fields_with_defaults(tmp_path, exclude):
    from mobilerun.sandbox.documents import build_document

    path = tmp_path / "routine.py"
    path.write_text(dedent(f"""\
        from pydantic import BaseModel, Field
        from mobilerun.sandbox import cloud

        class Input(BaseModel):
            pass

        class Output(BaseModel):
            image: str
            note: str = Field(default='', {exclude})

        @cloud('demo', files_folder='images', files={{'map.jpg': 'image'}},
               webhook='abcdef00-0000-4000-8000-000000000001')
        def task(value: Input) -> Output:
            raise AssertionError('entry point ran')
    """))
    with pytest.raises(SandboxDeclarationError, match="Webhook result objects"):
        build_document(package(path), "fn", 1, device=None)


@pytest.mark.parametrize("duck_type", ["annotation", "serializer"])
@pytest.mark.parametrize("expand", [True, False])
def test_webhook_rejects_duck_typed_serialization_only_on_expanded_branches(
    tmp_path, duck_type, expand
):
    from mobilerun.sandbox.documents import build_document

    path = tmp_path / "routine.py"
    source_field = "event.image" if expand else "image"
    event_type = "SerializeAsAny[Event]" if duck_type == "annotation" else "Event"
    serializer = "" if duck_type == "annotation" else dedent("""\
        @field_serializer('event')
        def serialize_event(self, event):
            return event.model_dump(serialize_as_any=True)
    """)
    path.write_text(dedent(f"""\
        from pydantic import BaseModel, SerializeAsAny, field_serializer
        from mobilerun.sandbox import cloud

        class Input(BaseModel):
            pass

        class Event(BaseModel):
            image: str

        class Output(BaseModel):
            image: str
            event: {event_type}

        @cloud('demo', files_folder='images', files={{'map.jpg': {source_field!r}}},
               webhook='abcdef00-0000-4000-8000-000000000001')
        def task(value: Input) -> Output:
            raise AssertionError('entry point ran')
    """).replace("\n@cloud", "\n" + indent(serializer, "    ") + "\n@cloud"))
    artifact = package(path)
    if expand:
        with pytest.raises(
            SandboxDeclarationError, match="custom or duck-typed serializers"
        ):
            build_document(artifact, "fn", 1, device=None)
    else:
        doc = build_document(artifact, "fn", 1, device=None)
        assert doc["steps"][-1]["with"]["data"] == {
            "result": {
                "image": {
                    "path": "/images/map.jpg",
                    "fileId": "{{steps.upload_1.body.fileId}}",
                },
                "event": "{{steps.function.body.result.event}}",
            }
        }


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

    from mobilerun.errors import BillingRequired
    from mobilerun.sandbox import deployment, packaging
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
    with pytest.raises(BillingRequired):
        await deployment.deploy_file(
            path, device_free=True, artifact=artifact, billing="no_commerce"
        )
    inspect.assert_not_called()


def test_schedule_inspects_real_model_before_cloud(tmp_path, monkeypatch):
    import asyncio

    from jsonschema import validate

    from mobilerun.sandbox import deployment
    from mobilerun.sandbox.errors import SandboxDeclarationError

    patch_sandbox(monkeypatch, "current_scope", lambda: pytest.fail("cloud access"))
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
        asyncio.run(deployment.deploy_file(source, device_free=True))
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
