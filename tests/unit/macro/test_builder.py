"""Versioned macros keep only replayable evidence."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from mobilerun.macro import builder
from mobilerun.macro.errors import MacroError
from mobilerun.models import Fingerprint
from tests.support.macro import write_steps


@pytest.fixture(autouse=True)
def macro_directory(tmp_path, monkeypatch):
    path = tmp_path / "macros"
    monkeypatch.setenv("MOBILERUN_MACRO_DIRECTORY", str(path))
    return path


def test_selected_export_omits_failed_masked_and_run_only_data(tmp_path):
    path = write_steps(
        tmp_path,
        {"i": 1},
        {"i": 2, "ok": False, "error": "rejected"},
        {"i": 3, "args": {"arguments": {"values": ["prefix ***"]}}},
        {"i": 4, "args": {"arguments": {"value": "<redacted>"}}},
        {"i": 5},
        {
            "i": 6,
            "action": "extension_call",
            "args": {"name": "vault.type_field", "arguments": {"name": "login"}},
            "note": "Type credential",
            "target": "login field",
            "screens": {"after": {"png": "screens/6.after.png"}},
        },
    )
    output = builder.build_macro(path, "routine", step_ids={2, 3, 4, 6, 1})

    rows = builder.load_macro(output)
    assert [row.i for row in rows] == [1, 6]
    assert rows[1].args == {
        "name": "vault.type_field",
        "arguments": {"name": "login"},
    }
    assert (rows[1].note, rows[1].target) == ("Type credential", "login field")
    assert (
        not {"started_at", "finished_at", "ok", "error", "by", "screens"}
        & rows[1].model_dump().keys()
    )


def test_latest_is_numeric_and_file_pins_version(tmp_path):
    first = builder.build_macro(
        write_steps(tmp_path, {"i": 1}), "routine", after_version=8
    )
    second = builder.build_macro(write_steps(tmp_path, {"i": 2}), "routine")
    assert (first.name, second.name) == ("v9.jsonl", "v10.jsonl")
    assert builder.load_macro("routine")[0].i == 2
    assert builder.load_macro(first.parent)[0].i == 2
    assert builder.load_macro(first)[0].i == 1


def test_fingerprints_with_null_identity_fields_round_trip(tmp_path):
    before = Fingerprint(app="settings", activity=None, elements=())
    after = Fingerprint(app=None, activity=".Done", elements=("id:done",))
    path = write_steps(
        tmp_path, {"before": before.model_dump(), "after": after.model_dump()}
    )

    (step,) = builder.load_macro(builder.build_macro(path, "routine"))

    assert step.before == before
    assert step.after == after


def test_racing_publishers_never_overwrite(tmp_path, monkeypatch):
    first_path = write_steps(tmp_path, {"i": 1})
    other = tmp_path / "other"
    other.mkdir()
    second_path = write_steps(other, {"i": 2})
    import os

    original = os.link
    ready = Barrier(2)

    def synchronize(source, target):
        if target.name == "v10.jsonl":
            ready.wait(timeout=5)
        original(source, target)

    monkeypatch.setattr(os, "link", synchronize)
    with ThreadPoolExecutor(max_workers=2) as pool:
        one = pool.submit(builder.build_macro, first_path, "routine", after_version=9)
        two = pool.submit(builder.build_macro, second_path, "routine", after_version=9)
        first, second = one.result(), two.result()

    assert {first.name, second.name} == {"v10.jsonl", "v11.jsonl"}
    assert builder.load_macro(first)[0].i == 1
    assert builder.load_macro(second)[0].i == 2
    assert len(list(first.parent.iterdir())) == 2


@pytest.mark.parametrize(
    "name", ["", "..", "../outside", "foo/bar", r"foo\bar", ".hidden"]
)
def test_unsafe_name_is_rejected(tmp_path, macro_directory, name):
    with pytest.raises(MacroError, match="name"):
        builder.build_macro(write_steps(tmp_path, {"i": 1}), name)
    assert not macro_directory.exists()


def test_empty_export_publishes_nothing(tmp_path, macro_directory):
    with pytest.raises(MacroError, match="No replayable"):
        builder.build_macro(write_steps(tmp_path, {"i": 1, "ok": False}), "empty")
    assert not macro_directory.exists()


def test_v2_is_not_a_macro_or_run(tmp_path):
    old = tmp_path / "macro.json"
    old.write_text('{"macro_schema_version": "2.0", "actions": []}')
    for target in (old, tmp_path):
        with pytest.raises(MacroError, match="Unsupported macro schema"):
            builder.load_macro(target)
    with pytest.raises(MacroError, match="Unsupported macro schema"):
        builder.build_macro(old, "routine")
