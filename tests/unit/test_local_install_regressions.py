import asyncio
import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from mobilerun.core.credentials import file as credentials
from mobilerun.mcp import installer
from mobilerun.skills import bundle


@pytest.mark.parametrize("private_temp", [False, True])
def test_windows_credentials_require_private_temp(tmp_path, monkeypatch, private_temp):
    windows = SimpleNamespace(**vars(os))
    windows.name = "nt"
    vars(windows).pop("fchmod", None)
    monkeypatch.setattr(credentials, "os", windows)
    created = []

    def private_file(parent):
        created.append(parent)
        if not private_temp:
            raise PermissionError("owner-only DACL unavailable")
        return credentials.tempfile.mkstemp(dir=parent)

    monkeypatch.setattr(
        credentials, "_windows_private_temp", private_file, raising=False
    )
    path = tmp_path / "credentials.yaml"
    store = credentials.FileCredentials(path)
    if not private_temp:
        with pytest.raises(PermissionError, match="DACL"):
            asyncio.run(store.save("pkg", "login", {"password": "first"}))
        assert created == [tmp_path] and not path.exists()
        return
    asyncio.run(store.save("pkg", "login", {"password": "first"}))
    path.chmod(0o666)  # Windows stat permissions do not represent ACLs.
    asyncio.run(store.update("pkg", "login", "password", "second"))
    assert store._read()["pkg"]["login"] == {"password": "second"}
    assert created == [tmp_path, tmp_path]


def test_dual_opencode_reports_both_and_refuses_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    original = '{"mcp": {"mobilerun": {"url":"one"}, /* keep */ "servers": {"mobilerun": {"url":"two"}}}}'
    path.write_text(original)
    assert (
        "both mcp.mobilerun and mcp.servers.mobilerun are installed"
        in installer.status("opencode", "project")
    )
    for operation in [
        installer.uninstall,
        lambda a, s: installer.install(a, s, force=True),
    ]:
        with pytest.raises(
            ValueError, match=r"both mcp\.mobilerun and mcp\.servers\.mobilerun"
        ):
            operation("opencode", "project")
        assert path.read_text() == original


@pytest.mark.skipif(os.name != "nt", reason="requires Windows DACL APIs")
def test_native_windows_shared_directory_does_not_expose_temp(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    subprocess.run(
        ["icacls.exe", str(shared), "/grant", "*S-1-1-0:(OI)(CI)R"],
        check=True,
        capture_output=True,
    )
    fd, path = credentials._windows_private_temp(shared)
    os.close(fd)
    env = dict(os.environ, MOBILERUN_TEST_TEMP=path)
    inspected = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'; $acl = Get-Acl -LiteralPath $env:MOBILERUN_TEST_TEMP; "
            "$acl.AreAccessRulesProtected; "
            "$acl.GetAccessRules($true,$true,[System.Security.Principal.SecurityIdentifier]) | "
            "ForEach-Object { $_.IdentityReference.Value }",
        ],
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )
    assert inspected.stdout.splitlines() == ["True", "S-1-3-4"]


def test_mixed_opencode_edits_only_legacy_entry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "custom")
    path = tmp_path / "opencode.jsonc"
    nested = '"servers": { /* keep */ "other": {"command": "ok"}}'
    path.write_text('{"mcp": {"custom": {"url": "old"}, ' + nested + "}}")
    assert installer.status("opencode", "project") == "installed"
    with pytest.raises(ValueError, match="--force"):
        installer.install("opencode", "project")
    assert installer.install("opencode", "project", force=True) == "installed"
    assert nested in path.read_text()
    assert installer.uninstall("opencode", "project") == "removed"
    assert nested in path.read_text() and '"custom"' not in path.read_text()


def test_codex_home_only_overrides_default_user_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "active"))
    assert installer.config_path("codex", "user") == tmp_path / "active/config.toml"
    assert installer.config_path("codex", "project") == tmp_path / ".codex/config.toml"
    monkeypatch.setenv(
        "MOBILERUN_AGENTS",
        json.dumps({"codex": {"skills": None, "mcp": ["other.toml", None]}}),
    )
    assert installer.config_path("codex", "user") == tmp_path / "other.toml"


def test_nested_name_can_match_unrelated_opencode_setting(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "timeout")
    path = tmp_path / "opencode.jsonc"
    path.write_text('{"mcp": {"timeout": 5, "servers": {"timeout": {"url": "old"}}}}')
    assert installer.uninstall("opencode", "project") == "removed"
    assert json.loads(path.read_text()) == {"mcp": {"timeout": 5, "servers": {}}}


@pytest.mark.parametrize("version", ["current", "retired", "old-manifest"])
def test_parent_uninstall_requires_all_managed_children(tmp_path, monkeypatch, version):
    root = tmp_path / "source"
    refs = root / "parent/references"
    refs.mkdir(parents=True)
    (refs.parent / "SKILL.md").write_text("parent")
    (refs / "child.md").write_text("reference")
    monkeypatch.setattr(bundle, "skills_root", lambda: root)
    target = tmp_path / "target"
    names = ["parent", "parent-child"]
    bundle.install(names, "codex", dir=target)
    if version == "retired":
        (refs / "child.md").unlink()  # A later release no longer ships this guide.
    elif version == "old-manifest":
        marker = target / "parent-child/.mobilerun-installed"
        manifest = json.loads(marker.read_text())
        del manifest["parent"]
        marker.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="parent-child"):
        bundle.uninstall(["parent"], "codex", dir=target, discard_changes=True)
    notes = target / "parent-child/notes.md"
    notes.write_text("keep")
    with pytest.raises(ValueError, match="local changes"):
        bundle.uninstall(names, "codex", dir=target)
    assert (target / "parent/SKILL.md").exists() and notes.exists()
    assert [
        r.status
        for r in bundle.uninstall(names, "codex", dir=target, discard_changes=True)
    ] == ["removed", "removed"]
