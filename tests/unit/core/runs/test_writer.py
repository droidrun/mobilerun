import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from stat import S_IMODE

import pytest


@pytest.fixture
def record():
    from mobilerun.models import DeviceInfo, RunRecord

    return RunRecord(
        run_id="run-1",
        kind="recording",
        device=DeviceInfo(platform="android", identifier="fake"),
        started_at=datetime.now(timezone.utc),
    )


def _step(i=1, ok=True):
    from mobilerun.models import RunStep

    return RunStep(
        i=i,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        action="tap",
        args={"x": 1, "y": 2},
        ok=ok,
        by="script",
    )


def test_writer_starts_lazily_commits_rows_and_atomically_finishes(record, tmp_path):
    from mobilerun.core.runs import RunWriter

    runs = tmp_path / "runs"
    writer = RunWriter(record, runs)
    assert writer.run_dir == runs / "run-1"
    assert writer.step_index == 0
    assert not runs.exists()
    writer.append(_step())
    manifest = writer.run_dir / "run.json"
    rows = writer.run_dir / "steps.jsonl"
    assert json.loads(manifest.read_text())["status"] == "running"
    assert json.loads(rows.read_text())["i"] == 1
    assert writer.step_index == 1
    writer.append(_step(2, ok=False))
    with manifest.open() as original:
        writer.finish("failed", failed_step=2)
        assert json.load(original)["status"] == "running"
    assert [json.loads(row)["ok"] for row in rows.read_text().splitlines()] == [
        True,
        False,
    ]
    final = json.loads(manifest.read_text())
    assert final["status"] == "failed" and final["failed_step"] == 2
    assert final["finished_at"].endswith("Z")
    for path, mode in (
        (runs, 0o700),
        (writer.run_dir, 0o700),
        (manifest, 0o600),
        (rows, 0o600),
    ):
        assert S_IMODE(path.stat().st_mode) == mode
    with pytest.raises(RuntimeError):
        writer.append(_step(3))


def test_finishing_an_unused_writer_leaves_no_files(record, tmp_path):
    from mobilerun.core.runs import RunWriter

    writer = RunWriter(record, tmp_path / "runs")
    writer.finish("cancelled")
    assert not writer.run_dir.exists()
    with pytest.raises(RuntimeError):
        writer.start()


def test_explicit_start_creates_empty_artifact_and_never_resumes(record, tmp_path):
    from mobilerun.core.runs import RunWriter

    writer = RunWriter(record, tmp_path)
    assert writer.start() == writer.start() == writer.run_dir
    assert (writer.run_dir / "steps.jsonl").read_bytes() == b""
    with pytest.raises(FileExistsError):
        RunWriter(record, tmp_path).append(_step())
    assert (writer.run_dir / "steps.jsonl").read_bytes() == b""


def test_writer_rejects_invalid_rows_before_touching_disk(record, tmp_path):
    from mobilerun.core.runs import RunWriter

    writer = RunWriter(record, tmp_path)
    with pytest.raises(ValueError):
        writer.append(_step(2))
    with pytest.raises(ValueError):
        writer.append(_step().model_copy(update={"args": {"x": float("nan")}}))
    with pytest.raises(ValueError):
        RunWriter(record.model_copy(update={"run_id": "../escape"}), tmp_path)
    assert not writer.run_dir.exists()
    assert writer.step_index == 0


def test_failed_append_preserves_index_and_prevents_further_writes(
    record, tmp_path, monkeypatch
):
    from mobilerun.core.runs import RunWriter
    from mobilerun.core.runs import writer as writer_module

    writer = RunWriter(record, tmp_path)
    writer.append(_step())
    path = writer.run_dir / "steps.jsonl"
    prior = path.read_bytes()

    def fail_open(file, flags, *args, **kwargs):
        assert Path(file) == path
        raise OSError("disk failure")

    with monkeypatch.context() as patch:
        patch.setattr(writer_module.os, "open", fail_open)
        with pytest.raises(OSError, match="disk failure"):
            writer.append(_step(2))
    assert writer.step_index == 1
    with pytest.raises(RuntimeError):
        writer.append(_step(2))
    assert path.read_bytes() == prior
    writer.finish("failed", failed_step=2)


def test_failed_finish_does_not_replace_manifest_or_leave_temp_file(
    record, tmp_path, monkeypatch
):
    from mobilerun.core.runs import RunWriter

    writer = RunWriter(record, tmp_path)
    writer.append(_step())

    def fail_replace(source, destination):
        raise OSError("replace failed")

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", fail_replace)
        with pytest.raises(OSError, match="replace failed"):
            writer.finish("succeeded")
    assert json.loads((writer.run_dir / "run.json").read_text())["status"] == "running"
    assert {p.name for p in writer.run_dir.iterdir()} == {"run.json", "steps.jsonl"}
    writer.finish("succeeded")


def test_recording_scope_restores_actor_and_cannot_override_veto():
    from mobilerun.core.runs import recording_policy, recording_scope

    assert recording_policy() == (True, None)
    with recording_scope(by="macro"):
        assert recording_policy() == (True, "macro")
        with recording_scope(enabled=False):
            assert recording_policy() == (False, "macro")
            with recording_scope(by="agent"):
                assert recording_policy() == (False, "agent")
        assert recording_policy() == (True, "macro")
    assert recording_policy() == (True, None)


def test_recording_scope_is_task_local_and_restores_after_error():
    from mobilerun.core.runs import recording_policy, recording_scope

    changed = asyncio.Event()
    read = asyncio.Event()

    async def suppress():
        with pytest.raises(ValueError), recording_scope(enabled=False):
            changed.set()
            await read.wait()
            assert recording_policy() == (False, None)
            raise ValueError
        assert recording_policy() == (True, None)

    async def exercise():
        task = asyncio.create_task(suppress())
        await changed.wait()
        try:
            assert recording_policy() == (True, None)
        finally:
            read.set()
            await task

    asyncio.run(exercise())
