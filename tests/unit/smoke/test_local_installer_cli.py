"""Exercise offline CLI installers without touching any developer config."""

import subprocess
import sys

from mobilerun.skills.bundle import list_skills


def test_local_config_and_skill_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "active"))

    def cli(*args, check=True):
        return subprocess.run(
            [sys.executable, "-m", "mobilerun", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
            check=check,
        )

    cli("mcp", "install", "--agent", "codex")
    assert (tmp_path / "active/config.toml").is_file()
    cli("mcp", "uninstall", "--agent", "codex")
    assert not (tmp_path / ".codex/config.toml").exists()

    skill = next(s for s in list_skills() if s.parent)
    options = ["--agent", "codex", "--dir", str(tmp_path / "skills")]
    cli("skills", "install", skill.name, *options)
    blocked = cli("skills", "uninstall", skill.parent, *options, check=False)
    assert blocked.returncode != 0 and skill.name in blocked.stdout + blocked.stderr
    cli("skills", "uninstall", skill.parent, skill.name, *options)
