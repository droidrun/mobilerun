"""Fresh-interpreter public imports and CLI startup boundaries."""

import itertools
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_EXPORTS = {
    "mobilerun": [
        "MobileAgent",
        "MobileAgentState",
        "load_llm",
        "ResultEvent",
        "DeviceDriver",
        "AndroidDriver",
        "RecordingDriver",
        "MacroPlayer",
        "replay_macro_file",
        "replay_macro_folder",
        "MobileConfig",
        "AgentConfig",
        "FastAgentConfig",
        "ManagerConfig",
        "ExecutorConfig",
        "AppCardConfig",
        "DeviceConfig",
        "LoggingConfig",
        "TracingConfig",
        "TelemetryConfig",
        "ToolsConfig",
        "CredentialsConfig",
        "LLMProfile",
    ],
    "mobilerun.agent": ["ResultEvent"],
    "mobilerun.agent.droid": ["MobileAgent", "MobileAgentState", "DroidAgent", "DroidAgentState"],
    "mobilerun.agent.utils": [
        "to_chat_messages",
        "has_content",
        "filter_empty_messages",
        "limit_history",
        "PromptResolver",
        "build_tool_registry",
        "Trajectory",
    ],
    "mobilerun.config_manager": [
        "MobileConfig",
        "DroidConfig",
        "LLMProfile",
        "AgentConfig",
        "FastAgentConfig",
        "ManagerConfig",
        "ExecutorConfig",
        "AppCardConfig",
        "DeviceConfig",
        "TelemetryConfig",
        "TracingConfig",
        "LoggingConfig",
        "ToolsConfig",
        "CredentialsConfig",
        "ConfigLoader",
        "OutdatedConfigError",
        "PathResolver",
        "PromptLoader",
    ],
}
BANNED = ("llama_index", "mobilerun.agent.droid", "mobilerun_sdk", "mobilerun_core", "segno")
IMPORTS = (
    "from mobilerun import MobileAgent",
    "from mobilerun import MobileConfig",
    "import mobilerun.config_manager",
    "import mobilerun; mobilerun.tools.AndroidDriver",
    "from mobilerun.agent import ResultEvent",
    "from mobilerun.agent.droid import MobileAgent",
    "import mobilerun.macro",
    "from mobilerun.cli.main import cli, run_command",
    "from droidrun import MobileAgent",
)


def fresh(code, *args, env=None):
    child_env = dict(os.environ)
    child_env.pop("MOBILERUN_CLOUD_API_KEY", None)
    child_env.update(
        MOBILERUN_TELEMETRY_ENABLED="false",
        NO_COLOR="1",
        PYTHONPATH=os.pathsep.join((str(ROOT / "compat"), str(ROOT))),
    )
    child_env.update(env or {})
    result = subprocess.run(
        [sys.executable, *args, "-c", code],
        cwd=ROOT,
        env=child_env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize(
    "module,name", [(mod, name) for mod, names in PUBLIC_EXPORTS.items() for name in names]
)
def test_public_export_imports_in_isolation(module, name):
    fresh(f"from {module} import {name}; assert {name} is not None")


@pytest.mark.parametrize("first,second", list(itertools.permutations(IMPORTS, 2)))
def test_public_import_order(first, second):
    fresh(first + "; " + second)


@pytest.mark.parametrize("module,names", PUBLIC_EXPORTS.items())
def test_public_names_and_dir_are_preserved(module, names):
    fresh(
        f"import importlib; m=importlib.import_module({module!r}); "
        f"assert m.__all__ == {names!r}; assert set({names!r}) <= set(dir(m))"
    )


@pytest.mark.parametrize(
    "name,canonical",
    [
        ("DroidAgent", "MobileAgent"),
        ("DroidAgentState", "MobileAgentState"),
        ("DroidConfig", "MobileConfig"),
    ],
)
def test_legacy_root_exports(name, canonical):
    fresh(
        f"import mobilerun; assert getattr(mobilerun,{name!r}) is getattr(mobilerun,{canonical!r})"
    )


@pytest.mark.parametrize(
    "args", [[], ["--help"], ["--version"], ["device", "--help"], ["completion", "zsh"]]
)
def test_cli_fast_import_boundary(args):
    fresh(f"""import sys
from click.testing import CliRunner
from mobilerun.cli.main import cli, run_command
if {args!r}:
    result=CliRunner().invoke(cli,{args!r})
    assert result.exit_code == 0,result.output
loaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in {BANNED!r})]
assert not loaded,loaded
""")


def test_shell_completion_fast_import_boundary():
    fresh(
        f"""import sys
from mobilerun.cli.main import cli
try:
    cli.main(prog_name='mobilerun',args=[])
except SystemExit as exc:
    assert exc.code == 0,exc.code
loaded=[name for name in sys.modules if any(name==p or name.startswith(p+'.') for p in {BANNED!r})]
assert not loaded,loaded
""",
        env={
            "_MOBILERUN_COMPLETE": "bash_complete",
            "COMP_WORDS": "mobilerun device l",
            "COMP_CWORD": "2",
        },
    )


@pytest.mark.parametrize("module", ["mobilerun", "droidrun", "mobilerun.macro", "droidrun.macro"])
def test_module_entrypoint_help(module):
    fresh(
        f"import runpy,sys; sys.argv=[{module!r},'--help']; runpy.run_module({module!r},run_name='__main__')"
    )


def test_compat_cli_entrypoint():
    fresh("import sys; from droidrun.cli_shim import main; sys.argv=['droidrun','--help']; main()")
