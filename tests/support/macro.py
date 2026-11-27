"""Offline macro replay device and JSONL fixtures."""

import json
from pathlib import Path

from mobilerun.models import RunStep
from tests.support.devices import MemoryDevice


class FakeDevice(MemoryDevice):
    state = None

    def __init__(self, *states):
        super().__init__()
        self.taps = []
        self.states = list(states or ({"phone_state": {"packageName": "other"}},))
        self.observations = 0

    async def tap(self, x, y):
        self.taps.append((x, y))

    async def _get_ui_tree(self):
        self.observations += 1
        if len(self.states) > 1:
            return self.states.pop(0)
        return self.states[0]


def write_macro(tmp_path: Path, guarded: bool = False, *, rows=None) -> Path:
    """Write a pinned macro; pass rows to override the default tap."""
    if rows is None:
        row = {"i": 1, "action": "tap", "args": {"x": 10, "y": 20}}
        if guarded:
            row["before"] = {"app": "expected", "activity": None, "elements": []}
        rows = [row]
    return _write_jsonl(tmp_path / "v1.jsonl", rows)


def write_steps(tmp_path: Path, *rows) -> Path:
    """Write run rows with defaults that individual tests can override."""
    return _write_jsonl(
        tmp_path / "steps.jsonl",
        (
            {
                "i": i,
                "started_at": "2026-11-27T00:00:00Z",
                "finished_at": "2026-11-27T00:00:01Z",
                "action": "tap",
                "args": {"x": 10, "y": 20},
                "ok": True,
                "by": "script",
                **row,
            }
            for i, row in enumerate(rows, 1)
        ),
    )


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_run_steps(run_dir: Path | None) -> list[RunStep]:
    """Read a recorder's persisted rows, requiring that it started a run."""
    assert run_dir is not None
    return [
        RunStep.model_validate_json(line)
        for line in (run_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
