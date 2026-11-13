"""Local sandbox command contracts without a sandbox dependency."""

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from mobilerun.cli import deviceauth, script
from mobilerun.cli.main import cli
from mobilerun.config_manager import ConfigLoader


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    from mobilerun_core import detect

    monkeypatch.setattr(detect, "adb_serial_present", lambda serial: False)
    monkeypatch.setattr(
        ConfigLoader, "get_user_config_path", lambda: tmp_path / "config.yaml"
    )
    monkeypatch.delenv("MOBILERUN_CONFIG", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    for name in (
        "MOBILERUN_DEVICE",
        "MOBILERUN_TARGET",
        "MOBILERUN_API_KEY",
        "MOBILERUN_CLOUD_API_KEY",
        "MOBILERUN_API_BASE_URL",
        "MOBILERUN_ANDROID_PORTAL_TOKEN",
        "MOBILERUN_IOS_PORTAL_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(deviceauth, "load_credentials", lambda: None)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("# sandbox script\n")
    return str(path)


@pytest.mark.parametrize(
    "explicit,venv,platform,expected",
    [
        ("/custom/python", "/project/.venv", "linux", "/custom/python"),
        (None, "/project/.venv", "linux", "/project/.venv/bin/python"),
        (None, "/project/.venv", "win32", "/project/.venv/Scripts/python.exe"),
        (None, None, "linux", sys.executable),
    ],
)
def test_interpreter_precedence(monkeypatch, explicit, venv, platform, expected):
    if venv:
        monkeypatch.setenv("VIRTUAL_ENV", venv)
    monkeypatch.setattr(script.sys, "platform", platform)
    call = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(script.subprocess, "run", call)
    assert script._interpreter(explicit) == expected
    assert call.call_args.args[0] == [expected, "-c", "import mobilerun_routines"]


@pytest.mark.parametrize("command", ["init", "check", "run"])
def test_missing_package(monkeypatch, source, command):
    call = Mock(return_value=subprocess.CompletedProcess([], 1))
    monkeypatch.setattr(script.subprocess, "run", call)
    args = ["script", command, source, "--python", "/custom/python"]
    if command == "run":
        args.extend(["--local", "-t", "cloud", "-d", CLOUD_ID, "--input", "{}"])
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 1, result.output
    assert result.stdout == ""
    assert result.stderr == (
        "mobilerun-routines is not installed in /custom/python. "
        'Install it with: pip install "mobilerun[routines]"\n'
    )
    assert call.call_count == 1


@pytest.mark.parametrize("code", [0, 3, 7])
def test_init_passthrough(monkeypatch, tmp_path, code):
    call = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], code, "created\n", "diagnostic\n"),
        ]
    )
    monkeypatch.setattr(script.subprocess, "run", call)
    path = str(tmp_path / "new.py")
    result = CliRunner().invoke(cli, ["script", "init", path])
    assert result.exit_code == code, result.output
    assert result.stdout == "created\n"
    assert result.stderr == "diagnostic\n"
    assert call.call_args.args[0] == [
        sys.executable,
        "-m",
        "mobilerun_routines",
        "new",
        path,
    ]


INSPECTION = {
    "ok": True,
    "entrypoint": "main",
    "async": True,
    "inputSchema": {"type": "object"},
    "outputSchema": {"type": "string"},
    "outputKind": "value",
    "datasetKey": None,
    "sdkVersion": "1.0",
}


@pytest.mark.parametrize("flags", [[], ["-j"]])
def test_check_output(monkeypatch, source, flags):
    raw = json.dumps(INSPECTION, indent=2) + "\n"
    call = Mock(
        side_effect=[
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0, raw, ""),
        ]
    )
    monkeypatch.setattr(script.subprocess, "run", call)
    result = CliRunner().invoke(cli, ["script", "check", source, *flags])
    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    if flags:
        assert result.stdout == raw
    else:
        assert (
            "Entry point: main\nOutput kind: value\nDataset key: None\n"
            in result.stdout
        )
        assert 'Input schema:\n{\n  "type": "object"\n}' in result.stdout
        assert 'Output schema:\n{\n  "type": "string"\n}' in result.stdout
    assert call.call_args.args[0][-2:] == ["inspect", source]


@pytest.mark.parametrize("code", [3, 8])
def test_check_error_passthrough(monkeypatch, source, code):
    monkeypatch.setattr(
        script.subprocess,
        "run",
        Mock(
            side_effect=[
                subprocess.CompletedProcess([], 0),
                subprocess.CompletedProcess([], code, "", "invalid entry point\n"),
            ]
        ),
    )
    result = CliRunner().invoke(cli, ["script", "check", source, "-j"])
    assert result.exit_code == code
    assert result.stdout == ""
    assert result.stderr == "invalid entry point\n"


@pytest.mark.parametrize(
    "args,message",
    [
        (["--input", "{}"], "--local"),
        (["--local"], "exactly one"),
        (["--local", "--input", "{"], "--input"),
    ],
)
def test_run_usage_errors(monkeypatch, source, args, message):
    call = Mock(side_effect=AssertionError("must validate before starting Python"))
    monkeypatch.setattr(script.subprocess, "run", call)
    result = CliRunner().invoke(cli, ["script", "run", source, *args])
    assert result.exit_code == 2, result.output
    assert message in result.stderr
    call.assert_not_called()


@pytest.mark.parametrize("code", [0, 5])
@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize("flags", [[], ["-j"]])
def test_run_result_and_cleanup(monkeypatch, source, tmp_path, code, from_file, flags):
    temporary_paths = []
    envelope = {"ok": code == 0, "result": {"count": 2}}
    raw = json.dumps(envelope) + "\n"
    state = str(tmp_path / "state.json")

    def run(command, **kwargs):
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0)
        assert command[:5] == [sys.executable, "-m", "mobilerun_routines", "run", source]  # fmt: skip
        input_path = Path(command[command.index("--input-file") + 1])
        result_path = Path(command[command.index("--result-file") + 1])
        temporary_paths.append(input_path.parent)
        assert json.loads(input_path.read_text()) == {"name": "example"}
        assert "--device-target" not in command
        assert kwargs["env"] is None
        assert command[-2:] == ["--state-file", state]
        assert "stderr" not in kwargs and not kwargs.get("capture_output")
        result_path.write_text(raw)
        return subprocess.CompletedProcess(command, code, "")

    monkeypatch.setattr(script.subprocess, "run", run)
    input_path = tmp_path / "source-input.json"
    input_path.write_text('{"name": "example"}')
    inputs = (
        ["--input-file", str(input_path)]
        if from_file
        else ["--input", input_path.read_text()]
    )
    result = CliRunner().invoke(
        cli,
        ["script", "run", source, "--local", *inputs, "--state-file", state, *flags],
    )
    assert result.exit_code == code, result.output
    assert json.loads(result.stdout) == envelope
    assert result.stdout == (raw if flags else json.dumps(envelope, indent=2) + "\n")
    assert temporary_paths and all(not path.exists() for path in temporary_paths)
    assert input_path.exists()


@pytest.mark.parametrize("mode", ["failure", "exception"])
def test_run_cleanup_without_result(monkeypatch, source, mode):
    temporary_paths = []

    def run(command, **kwargs):
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0)
        temporary_paths.append(Path(command[command.index("--input-file") + 1]).parent)
        if mode == "exception":
            raise OSError("cannot launch interpreter")
        return subprocess.CompletedProcess(command, 9, "")

    monkeypatch.setattr(script.subprocess, "run", run)
    result = CliRunner().invoke(
        cli, ["script", "run", source, "--local", "--input", "null"]
    )
    assert result.exit_code == (1 if mode == "exception" else 9), result.output
    assert result.stdout == ""
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_run_rejects_both_inputs(monkeypatch, source):
    call = Mock()
    monkeypatch.setattr(script.subprocess, "run", call)
    result = CliRunner().invoke(
        cli,
        [
            "script",
            "run",
            source,
            "--local",
            "--input",
            "{}",
            "--input-file",
            source,
        ],
    )
    assert result.exit_code == 2
    assert "exactly one" in result.stderr
    call.assert_not_called()


CLOUD_ID = "00000000-0000-4000-8000-000000000000"


@pytest.mark.parametrize(
    "flags,device_args",
    [
        (
            ["-d", "emulator-5554"],
            [
                "--device-target",
                "local",
                "--device-platform",
                "android",
                "--device-id",
                "emulator-5554",
            ],
        ),
        (
            ["-P", "android", "-d", "http://127.0.0.1:8080"],
            [
                "--device-target",
                "local",
                "--device-platform",
                "android",
                "--device-url",
                "http://127.0.0.1:8080",
            ],
        ),
        (
            ["-P", "ios", "-d", "http://127.0.0.1:6643/"],
            [
                "--device-target",
                "local",
                "--device-platform",
                "ios",
                "--device-url",
                "http://127.0.0.1:6643",
            ],
        ),
    ],
)
def test_run_local_device_mapping(monkeypatch, source, flags, device_args):
    config_token = "unit-config-portal-token"
    config_only_token = "--device-url" in device_args and "android" in device_args
    if config_only_token:
        config = ConfigLoader.load()
        config.device.auth_token = config_token
        monkeypatch.setattr(ConfigLoader, "load", lambda config_path=None: config)

    def run(command, **kwargs):
        assert config_token not in " ".join(command)
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0)
        assert command[-len(device_args) :] == device_args
        if config_only_token:
            assert kwargs["env"]["MOBILERUN_ANDROID_PORTAL_TOKEN"] == config_token
        else:
            assert kwargs["env"] is None
        Path(command[command.index("--result-file") + 1]).write_text('{"ok": true}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", run)
    result = CliRunner().invoke(
        cli, ["script", "run", source, "--local", "--input", "{}", *flags]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    assert config_token not in result.output


@pytest.mark.parametrize(
    "flags", [["-t", "cloud", "-d", CLOUD_ID], ["-d", CLOUD_ID], []]
)
def test_run_cloud_device(monkeypatch, source, flags):
    if not flags:
        monkeypatch.setenv("MOBILERUN_DEVICE", CLOUD_ID)
        monkeypatch.setenv("MOBILERUN_TARGET", "cloud")
    monkeypatch.setenv("MOBILERUN_API_KEY", "unit-script-env-key")
    monkeypatch.setenv("SCRIPT_CUSTOM_VAR", "inherited")

    def run(command, **kwargs):
        assert "unit-script-env-key" not in " ".join(command)
        if "-c" in command:
            return subprocess.CompletedProcess(command, 0)
        assert command[-4:] == ["--device-target", "cloud", "--device-id", CLOUD_ID]
        assert kwargs["env"]["MOBILERUN_API_KEY"] == "unit-script-env-key"
        assert kwargs["env"]["MOBILERUN_API_BASE_URL"] == "https://api.example.com/v1"
        assert kwargs["env"]["SCRIPT_CUSTOM_VAR"] == "inherited"
        Path(command[command.index("--result-file") + 1]).write_text('{"ok": true}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(script.subprocess, "run", run)
    result = CliRunner().invoke(
        cli,
        [
            "script",
            "run",
            source,
            "--local",
            "--input",
            "{}",
            "--api-url",
            "https://api.example.com/v1",
            *flags,
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"ok": True}
    assert "unit-script-env-key" not in result.output


@pytest.mark.parametrize(
    "flags,message",
    [
        (["-t", "cloud", "-d", CLOUD_ID], "No cloud API key found"),
        (["-t", "cloud"], "Cloud device required"),
        (
            ["-d", "http://127.0.0.1:8080"],
            "Portal URL requires MOBILERUN_ANDROID_PORTAL_TOKEN",
        ),
    ],
)
def test_run_device_error_does_not_start_routines(monkeypatch, source, flags, message):
    call = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(script.subprocess, "run", call)
    result = CliRunner().invoke(
        cli, ["script", "run", source, "--local", "--input", "{}", *flags]
    )
    assert result.exit_code == 1, result.output
    assert message in result.stderr
    if "-d" in flags and CLOUD_ID in flags:
        assert "mobilerun auth login" in result.stderr
    call.assert_called_once()
    assert call.call_args.args[0] == [sys.executable, "-c", "import mobilerun_routines"]
