"""Sandbox deployment document contracts."""

import pytest

from mobilerun.sandbox.models import Artifact, CloudDeclaration


def artifact(**options):
    return Artifact(
        CloudDeclaration("hello", **options),
        "source",
        "digest",
        {"type": "object"},
        None,
    )


def test_build_endpoint_document_uses_pinned_version_and_no_device():
    from mobilerun.sandbox.deployment import build_document

    doc = build_document(artifact(endpoint=True), "function", 3, device=None)
    assert doc["on"] == {"custom": {"inputs": {"type": "object"}}}
    assert "devices" not in doc
    assert doc["steps"][0]["with"]["version"] == 3


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ("", {"custom": {"inputs": {"type": "object"}}}),
        (", endpoint=False", None),
        (
            ", schedule='0 * * * *'",
            {"schedule": {"cron": "0 * * * *", "timezone": "UTC"}},
        ),
    ],
)
def test_literal_declaration_flow_resolution(tmp_path, options, expected):
    from mobilerun.sandbox.deployment import build_document
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        f"@cloud('hello'{options})\ndef hello(): pass\n"
    )
    packaged = Artifact(
        read_declaration(path), "source", "digest", {"type": "object"}, None
    )
    doc = build_document(packaged, "function", 1, device=None)
    assert (doc["on"] if doc is not None else None) == expected


@pytest.mark.parametrize(
    "options",
    [
        "schedule='0 * * * *', endpoint=True",
        "endpoint=1",
    ],
)
def test_literal_declaration_rejects_invalid_trigger_options(tmp_path, options):
    from mobilerun.sandbox.errors import SandboxDeclarationError
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        f"@cloud('hello', {options})\ndef hello(): pass\n"
    )
    with pytest.raises(SandboxDeclarationError):
        read_declaration(path)


def test_literal_declaration_rejects_removed_manual_trigger(tmp_path):
    from mobilerun.sandbox.errors import SandboxDeclarationError
    from mobilerun.sandbox.packaging import read_declaration

    path = tmp_path / "main.py"
    path.write_text(
        "from mobilerun.sandbox import cloud\n"
        "@cloud('hello', trigger='manual')\ndef hello(): pass\n"
    )
    with pytest.raises(
        SandboxDeclarationError, match=r"manual.*removed.*endpoint.*device="
    ):
        read_declaration(path)


def test_invalid_cron_fails_locally():
    from mobilerun.sandbox.deployment import build_document
    from mobilerun.sandbox.errors import SandboxDeclarationError

    with pytest.raises(SandboxDeclarationError):
        build_document(artifact(schedule="bad"), "function", 1, device=None)


@pytest.mark.parametrize(
    "declaration", [CloudDeclaration(""), CloudDeclaration("x", endpoint="yes")]
)
def test_invalid_declaration_is_rejected(declaration):
    from mobilerun.sandbox.deployment import build_document
    from mobilerun.sandbox.errors import SandboxDeclarationError

    with pytest.raises(SandboxDeclarationError):
        build_document(
            Artifact(declaration, "source", "digest", {}, None), "f", 1, device=None
        )


def test_equivalence_detects_device_removal_and_schema_changes():
    from mobilerun.sandbox.deployment import equivalent_document

    assert not equivalent_document(
        {"on": {"custom": {}}, "devices": ["phone"]}, {"on": {"custom": {}}}
    )
    assert not equivalent_document(
        {"on": {"custom": {"inputs": {}}}}, {"on": {"custom": {}}}
    )
    assert equivalent_document({"name": "x", "server": 1}, {"name": "x"})
