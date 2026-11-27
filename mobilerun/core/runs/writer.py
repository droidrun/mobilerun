"""Private append-only run artifacts and task-local recording policy."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from mobilerun.models import Actor, RunRecord, RunStatus, RunStep

_policy: ContextVar[tuple[bool, Actor | None]] = ContextVar(
    "run_recording_policy", default=(True, None)
)


def recording_policy() -> tuple[bool, Actor | None]:
    """Read (enabled, actor override) for the current task."""
    return _policy.get()


@contextmanager
def recording_scope(*, enabled: bool = True, by: Actor | None = None) -> Iterator[None]:
    """Override the actor or suppress recording without lifting a parent veto."""
    parent_enabled, parent_actor = recording_policy()
    token = _policy.set(
        (parent_enabled and enabled, by if by is not None else parent_actor)
    )
    try:
        yield
    finally:
        _policy.reset(token)


@dataclass
class _StepCollection:
    ids: list[int] = field(default_factory=list)
    active: bool = True


class RunWriter:
    """Write one new run, without reopening, resuming or owning the device."""

    def __init__(
        self, record: RunRecord | Callable[[], RunRecord], runs_dir: Path
    ) -> None:
        self._factory = record if callable(record) else lambda: record
        self._record: RunRecord | None = None
        self._runs_dir = runs_dir
        if isinstance(record, RunRecord):
            self._get_record()
        self._step_index = 0
        self._started = False
        self._finished = False
        self._write_failed = False
        self._collectors: ContextVar[tuple[_StepCollection, ...]] = ContextVar(
            "run_step_collectors", default=()
        )

    def _get_record(self) -> RunRecord:
        if self._record is not None:
            return self._record
        record = RunRecord.model_validate(self._factory())
        if (
            record.status != "running"
            or record.finished_at is not None
            or record.failed_step is not None
        ):
            raise ValueError("A new run requires an unfinished, running record")
        self._record = record
        return record

    @property
    def run_dir(self) -> Path:
        """Return the artifact path, even before it exists."""
        return self._runs_dir / self._get_record().run_id

    @property
    def started_dir(self) -> Path | None:
        return self.run_dir if self._started else None

    @property
    def step_index(self) -> int:
        """Return the last committed row index (0 until the first append)."""
        return self._step_index

    @contextmanager
    def collect_steps(self) -> Iterator[list[int]]:
        """Collect committed rows in this scope and child tasks that inherit it."""
        collection = _StepCollection()
        token = self._collectors.set((*self._collectors.get(), collection))
        try:
            yield collection.ids
        finally:
            collection.active = False
            self._collectors.reset(token)

    def start(self) -> Path:
        """Create the run before its first action, if another sink needs its path."""
        if self._finished or self._write_failed:
            raise RuntimeError("Run is closed for recording")
        if not self._started:
            try:
                self.run_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                self.run_dir.mkdir(mode=0o700)
                os.close(
                    os.open(
                        self.run_dir / "steps.jsonl",
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                        0o600,
                    )
                )
                self._write_record(self._get_record())
            except OSError:
                self._write_failed = True
                raise
            self._started = True
        return self.run_dir

    def append(self, step: RunStep) -> None:
        """Validate and flush the next row before advancing the committed index."""
        step = RunStep.model_validate(step)
        if step.i != self.step_index + 1:
            raise ValueError("Run steps must be appended in consecutive 1-based order")
        row = json.dumps(step.model_dump(mode="json"), allow_nan=False) + "\n"
        path = self.start() / "steps.jsonl"
        try:
            with os.fdopen(os.open(path, os.O_APPEND | os.O_WRONLY), "ab") as file:
                file.write(row.encode("utf-8"))
                file.flush()
        except OSError:
            self._write_failed = True
            raise
        self._step_index = step.i
        for collection in self._collectors.get():
            if collection.active:
                collection.ids.append(step.i)

    def finish(self, status: RunStatus, failed_step: int | None = None) -> None:
        """Close the run with a terminal status, without starting an unused run."""
        if self._finished:
            return
        if status == "running":
            raise ValueError("A finished run requires a terminal status")
        if self._record is None:
            self._finished = True
            return
        record = RunRecord.model_validate(
            self._record.model_copy(
                update={
                    "status": status,
                    "failed_step": failed_step,
                    "finished_at": datetime.now(UTC),
                }
            )
        )
        if self._started:
            self._write_record(record)
        self._finished = True

    def _write_record(self, record: RunRecord) -> None:
        temp_path = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.run_dir, delete=False
            ) as file:
                temp_path = Path(file.name)
                file.write(record.model_dump_json() + "\n")
            os.replace(temp_path, self.run_dir / "run.json")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
