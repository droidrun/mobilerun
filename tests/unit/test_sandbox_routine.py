import json

import mobilerun_routines
import mobilerun_routines.context
import pytest
from mobilerun_routines.cli import main


def test_public_routine_api_is_reexported():
    import mobilerun.sandbox as sandbox

    for name in mobilerun_routines.__all__:
        assert getattr(sandbox, name) is getattr(mobilerun_routines, name)
    assert callable(sandbox.routine)


@pytest.mark.parametrize("async_prefix", ["", "async "])
def test_routine_matches_script_inspect_and_run(
    tmp_path, monkeypatch, capsys, async_prefix
):
    template = """\
from pydantic import BaseModel
from {module} import {decorator}, Context

class Input(BaseModel):
    message: str

class Output(BaseModel):
    value: str

@{decorator}
{async_prefix}def main(input: Input, ctx: Context) -> Output:
    return Output(value=ctx.device().echo(input.message))
"""
    paths = []
    for module, decorator in (
        ("mobilerun.sandbox", "routine"),
        ("mobilerun_routines", "script"),
    ):
        path = tmp_path / f"{decorator}.py"
        path.write_text(
            template.format(
                module=module, decorator=decorator, async_prefix=async_prefix
            )
        )
        paths.append(path)

    inspected = []
    for path in paths:
        assert main(["inspect", str(path)]) == 0
        inspected.append(json.loads(capsys.readouterr().out))
    assert inspected[0] == inspected[1]
    assert inspected[0]["ok"] is True

    class FakeDevice:
        def echo(self, message):
            return f"device: {message}"

    monkeypatch.setattr(
        mobilerun_routines.context, "connect_device", lambda device_id: FakeDevice()
    )
    input_file = tmp_path / "input.json"
    input_file.write_text('{"message":"hello"}')
    envelopes = []
    for path in paths:
        result_file = tmp_path / f"{path.stem}.json"
        assert (
            main(
                [
                    "run",
                    str(path),
                    "--input-file",
                    str(input_file),
                    "--result-file",
                    str(result_file),
                    "--device-id",
                    "fake",
                ]
            )
            == 0
        )
        envelopes.append(json.loads(result_file.read_text()))
    assert (
        envelopes
        == [{"ok": True, "outcome": "succeeded", "result": {"value": "device: hello"}}]
        * 2
    )
