"""Local MCP configuration and installer safety contracts."""

import json

import pytest

from mobilerun.mcp import installer, utils


def test_mcp_settings_overrides_are_read_per_call(tmp_path, monkeypatch):
    import json

    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_MCP_URL", "https://example.test/mcp")
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "custom-server")
    assert install("cursor", "user") == "installed"
    path = tmp_path / ".cursor/mcp.json"
    assert (
        json.loads(path.read_text())["mcpServers"]["custom-server"]["url"]
        == "https://example.test/mcp"
    )
    assert status("cursor", "user") == "installed"
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "other-server")
    assert status("cursor", "user") == "not installed"
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "custom-server")
    assert uninstall("cursor", "user") == "removed"


def test_mcp_empty_url_uses_default(tmp_path, monkeypatch):
    import json

    from mobilerun.mcp.installer import install
    from mobilerun.mcp.settings import get_mcp_settings

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_MCP_URL", "https://example.test/mcp")
    assert get_mcp_settings().MOBILERUN_MCP_URL == "https://example.test/mcp"
    monkeypatch.setenv("MOBILERUN_MCP_URL", "")
    install("cursor", "user")
    assert (
        json.loads((tmp_path / ".cursor/mcp.json").read_text())["mcpServers"][
            "mobilerun"
        ]["url"]
        == "https://cloud.mobilerun.ai/api/mcp"
    )


def test_mcp_cli_oauth_hint_uses_custom_server_name(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from mobilerun.mcp.cli.group import mcp

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "custom-server")
    result = CliRunner().invoke(mcp, ["install", "--agent", "codex"])
    assert result.exit_code == 0, result.output
    assert "codex mcp login custom-server" in result.output


def test_jsonc_custom_server_name_round_trip(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MOBILERUN_MCP_SERVER_NAME", "custom-server")
    path = tmp_path / "opencode.jsonc"
    original = '{\n  "mcp": {"servers": {\n    "other": {}, // keep\n  }}\n}\n'
    path.write_text(original)
    assert install("opencode", "project") == "installed"
    assert '"custom-server"' in path.read_text()
    assert status("opencode", "project") == "installed"
    assert uninstall("opencode", "project") == "removed"
    assert path.read_text() == original


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "cursor"])
@pytest.mark.parametrize("scope", ["user", "project"])
@pytest.mark.parametrize("api_key", [False, True])
def test_install_is_idempotent_and_uninstall_removes_server(
    tmp_path, monkeypatch, agent, scope, api_key
):
    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MOBILERUN_API_KEY", "never-write-this-secret")
    monkeypatch.chdir(tmp_path)
    if agent == "claude-code" and scope == "user" and api_key:
        with pytest.raises(ValueError, match="--project"):
            install(agent, scope, api_key=True)
        return
    assert status(agent, scope) == "not installed"
    assert install(agent, scope, api_key=api_key) == "installed"
    assert status(agent, scope) == "installed"
    assert install(agent, scope, api_key=api_key) == "exists"
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert b"never-write-this-secret" not in path.read_bytes()
    assert uninstall(agent, scope) == "removed"
    assert status(agent, scope) == "not installed"


@pytest.mark.parametrize("legacy", [False, True])
def test_opencode_version_layout(tmp_path, monkeypatch, legacy):
    import json

    from mobilerun.mcp.installer import install

    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".config/opencode/opencode.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "mcp": (
                    {"other": {"type": "remote", "url": "https://example.com"}}
                    if legacy
                    else {
                        "servers": {
                            "other": {"type": "remote", "url": "https://example.com"}
                        }
                    }
                )
            }
        )
    )
    install("opencode", "user")
    mcp = json.loads(path.read_text())["mcp"]
    servers = mcp if legacy else mcp["servers"]
    assert "other" in servers and "mobilerun" in servers


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "cursor"])
@pytest.mark.parametrize("scope", ["user", "project"])
def test_preserves_other_servers_and_requires_force(
    tmp_path, monkeypatch, agent, scope
):
    from mobilerun.mcp.installer import config_path, install, uninstall

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    path = config_path(agent, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    if agent == "codex":
        path.write_text(
            '# Keep this comment\n[mcp_servers.other]\nurl = "https://other"\n[mcp_servers.mobilerun]\nurl = "https://old"\n'
        )
    else:
        import json

        key = "mcp" if agent == "opencode" else "mcpServers"
        servers = {
            "other": {"url": "https://other"},
            "mobilerun": {"url": "https://old"},
        }
        path.write_text(
            json.dumps(
                {
                    key: {"servers": servers} if agent == "opencode" else servers,
                    "unrelated": 42,
                }
            )
        )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="--force"):
        install(agent, scope)
    assert path.read_bytes() == before
    assert install(agent, scope, force=True) == "installed"
    assert "https://other" in path.read_text()
    if agent == "codex":
        assert "# Keep this comment" in path.read_text()
    else:
        assert '"unrelated": 42' in path.read_text()
    assert uninstall(agent, scope) == "removed"
    assert "https://other" in path.read_text()
    assert "https://cloud.mobilerun.ai/api/mcp" not in path.read_text()


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "cursor"])
@pytest.mark.parametrize("scope", ["user", "project"])
def test_malformed_config_is_untouched(tmp_path, monkeypatch, agent, scope):
    from mobilerun.mcp.installer import config_path, install, status, uninstall

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    path = config_path(agent, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{{broken")
    for action in (install, uninstall, status):
        with pytest.raises(ValueError, match="Invalid config"):
            action(agent, scope)
        assert path.read_text() == "{{broken"


def test_cli_status_and_noninteractive(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from mobilerun.cli.app import cli
    from mobilerun.mcp.cli import group

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(group, "is_interactive", lambda: False)
    runner = CliRunner()
    assert "--agent" in runner.invoke(cli, ["mcp", "install"]).output
    result = runner.invoke(cli, ["mcp", "install", "--agent", "codex", "--project"])
    assert result.exit_code == 0, result.output
    assert "trust" in result.output
    result = runner.invoke(cli, ["mcp", "list"])
    assert result.exit_code == 0, result.output
    assert "codex (project): installed" in result.output
    assert "codex (user): not installed" in result.output


def test_cli_interactive_agent_picker(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from mobilerun.mcp.cli import group

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(group, "is_interactive", lambda: True)
    monkeypatch.setattr(group, "select_prompt", lambda message, choices: "cursor")
    result = CliRunner().invoke(group.mcp, ["install"])
    assert result.exit_code == 0, result.output
    assert "cursor (user): installed" in result.output


@pytest.mark.parametrize(
    "agent,next_step",
    [
        ("opencode", "Next: sign in with `opencode mcp auth mobilerun`"),
        ("codex", "Next: sign in with `codex mcp login mobilerun`"),
        (
            "claude-code",
            "Next: sign in from Claude Code: run /mcp, select mobilerun, then Authenticate",
        ),
        (
            "cursor",
            "Next: sign in from Cursor: open its MCP server settings, select mobilerun, then sign in",
        ),
    ],
)
@pytest.mark.parametrize("install_result", ["installed", "exists"])
def test_cli_oauth_install_prints_agent_sign_in_next_step(
    tmp_path, monkeypatch, agent, next_step, install_result
):
    from click.testing import CliRunner

    from mobilerun.mcp.cli import group

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        group.installer, "install", lambda *args, **kwargs: install_result
    )
    result = CliRunner().invoke(group.mcp, ["install", "--agent", agent])

    assert result.exit_code == 0, result.output
    assert next_step in result.output


@pytest.mark.parametrize("agent", ["opencode", "codex", "claude-code", "cursor"])
def test_cli_api_key_install_prints_environment_reminder(tmp_path, monkeypatch, agent):
    from click.testing import CliRunner

    from mobilerun.mcp.cli import group

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(group.installer, "install", lambda *args, **kwargs: "installed")
    result = CliRunner().invoke(group.mcp, ["install", "--agent", agent, "--api-key"])

    assert result.exit_code == 0, result.output
    assert "MOBILERUN_API_KEY" in result.output
    assert "Next: sign in" not in result.output


@pytest.mark.parametrize(
    "agent,reference",
    [
        ("claude-code", "${MOBILERUN_API_KEY}"),
        ("codex", "MOBILERUN_API_KEY"),
        ("opencode", "{env:MOBILERUN_API_KEY}"),
        ("cursor", "${env:MOBILERUN_API_KEY}"),
    ],
)
def test_api_key_reference(tmp_path, monkeypatch, agent, reference):
    from mobilerun.mcp.installer import config_path, install

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    install(agent, "project", api_key=True)
    assert reference in config_path(agent, "project").read_text()


def test_opencode_timeout_only_is_v2(tmp_path, monkeypatch):
    import json

    from mobilerun.mcp.installer import install

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.json"
    path.write_text(json.dumps({"mcp": {"timeout": {"connect": 200}}}))
    install("opencode", "project")
    mcp = json.loads(path.read_text())["mcp"]
    assert mcp["timeout"] == {"connect": 200}
    assert "mobilerun" in mcp["servers"]
    assert "mobilerun" not in {k: v for k, v in mcp.items() if k != "servers"}


@pytest.mark.parametrize("scope", ["user", "project"])
def test_opencode_jsonc_plain_json_used_for_all_operations(
    tmp_path, monkeypatch, scope
):
    import json

    from mobilerun.mcp.installer import config_path, install, status, uninstall

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    path = config_path("opencode", scope).with_suffix(".jsonc")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"other": true}')
    assert install("opencode", scope) == "installed"
    assert status("opencode", scope) == "installed"
    assert json.loads(path.read_text())["other"] is True
    assert not path.with_suffix(".json").exists()
    assert uninstall("opencode", scope) == "removed"
    assert status("opencode", scope) == "not installed"


def test_opencode_jsonc_comments_round_trip(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    text = '{\n  // settings\n  "mcp": {\n    "servers": {\n      "other": {"url": "https://example/*x*/"}, // keep\n    },\n  },\n}\n'
    path.write_text(text)
    assert install("opencode", "project") == "installed"
    assert status("opencode", "project") == "installed"
    assert "// settings" in path.read_text()
    assert uninstall("opencode", "project") == "removed"
    assert path.read_text() == text


@pytest.mark.parametrize(
    "text,keys",
    [
        ('{ /* root */ "other": true, }', ["mcp", "servers"]),
        ('{"mcp": { /* settings */ "timeout": 5, }}', ["mcp", "servers"]),
        ('{"mcp": {"old": {"command": "tool"},}}', ["mcp"]),
        ('{"mcp": {"servers": {}}}', ["mcp", "servers"]),
    ],
)
def test_opencode_jsonc_layouts(tmp_path, monkeypatch, text, keys):
    from mobilerun.mcp.installer import install, uninstall
    from mobilerun.mcp.utils import _loads

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    path.write_text(text)
    assert install("opencode", "project") == "installed"
    data = _loads(path.read_text())
    for key in keys:
        data = data[key]
    assert "mobilerun" in data
    assert uninstall("opencode", "project") == "removed"
    assert "mobilerun" not in str(_loads(path.read_text()))


def test_opencode_jsonc_force_and_escaped_strings(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, uninstall
    from mobilerun.mcp.utils import _loads

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    text = '{\n  "mcp": {"servers": {\n    "other": {"url": "// /* } \\""}, // keep\n    "mobilerun": {"url": "old"},\n  }},\n}\n'
    path.write_text(text)
    with pytest.raises(ValueError, match="--force"):
        install("opencode", "project")
    assert path.read_text() == text
    assert install("opencode", "project", force=True) == "installed"
    assert _loads(path.read_text())["mcp"]["servers"]["other"]["url"] == '// /* } "'
    assert "// keep" in path.read_text()
    assert uninstall("opencode", "project") == "removed"
    assert "// keep" in path.read_text()


def test_opencode_jsonc_round_trip_without_trailing_comma(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    text = '{\n  "mcp": {\n    "servers": {\n      "other": {} // keep\n    }\n  }\n}\n'
    path.write_text(text)
    install("opencode", "project")
    uninstall("opencode", "project")
    assert path.read_text() == text


@pytest.mark.parametrize(
    "text",
    [
        '{"mcp":{"servers":{"mobilerun":{},"mobilerun":{}}}}',
        '{"mcp":{},"mcp":{"servers":{"mobilerun":{}}}}',
        '{"mcp":{"servers":{},"servers":{"mobilerun":{}}}}',
    ],
)
def test_jsonc_duplicate_target_is_rejected(tmp_path, monkeypatch, text):
    from mobilerun.mcp.installer import install, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    path.write_text(text)
    for action in (
        lambda: install("opencode", "project", force=True),
        lambda: uninstall("opencode", "project"),
    ):
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            action()
        assert path.read_text() == text


@pytest.mark.parametrize("bad", ["[,]", "{,}", "[1,,2]", "[1,,]"])
def test_jsonc_malformed_commas_rejected(tmp_path, monkeypatch, bad):
    from mobilerun.mcp.installer import install

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    content = ('{"other":' + bad + "}").encode()
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Invalid config") as exc:
        install("opencode", "project")
    assert type(exc.value) is ValueError
    assert path.read_bytes() == content


def test_jsonc_crlf_insert_replace_remove(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    original = b'{\r\n  "mcp": {\r\n    "servers": {\r\n      "other": {}\r\n    }\r\n  }\r\n}\r\n'
    path.write_bytes(original)
    install("opencode", "project")
    inserted = path.read_bytes()
    assert b'"mobilerun"' in inserted
    assert b"\n" not in inserted.replace(b"\r\n", b"")
    install("opencode", "project", api_key=True, force=True)
    replaced = path.read_bytes()
    assert b"Authorization" in replaced
    assert b"\n" not in replaced.replace(b"\r\n", b"")
    uninstall("opencode", "project")
    assert path.read_bytes() == original


def test_jsonc_dangling_symlink(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, status

    monkeypatch.chdir(tmp_path)
    link = tmp_path / "opencode.jsonc"
    target = tmp_path / "nested" / "target.jsonc"
    link.symlink_to(target)
    assert install("opencode", "project") == "installed"
    assert link.is_symlink() and target.exists()
    assert status("opencode", "project") == "installed"


def test_jsonc_bom_is_preserved(tmp_path, monkeypatch):
    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    original = b'\xef\xbb\xbf{\n  "mcp": {"servers": {\n  }}\n}\n'
    path.write_bytes(original)
    install("opencode", "project")
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert status("opencode", "project") == "installed"
    uninstall("opencode", "project")
    assert path.read_bytes() == original


@pytest.mark.parametrize("text", ['{"x": ,}', "{/* unterminated"])
def test_opencode_invalid_jsonc_is_safe(tmp_path, monkeypatch, text):
    from mobilerun.mcp.installer import install, status, uninstall

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "opencode.jsonc"
    path.write_text(text)
    for action in (install, status, uninstall):
        with pytest.raises(ValueError, match=r"Invalid config .*opencode.jsonc"):
            action("opencode", "project")
        assert path.read_text() == text
    assert not (tmp_path / "opencode.json").exists()


def test_symlinked_config_preserved(tmp_path, monkeypatch):
    import json

    from mobilerun.mcp.installer import install, uninstall

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    target = tmp_path / "dotfiles" / "claude.json"
    target.parent.mkdir()
    target.write_text('{"other": true}')
    link = home / ".claude.json"
    link.symlink_to(target)
    install("claude-code", "user")
    assert link.is_symlink()
    assert "mobilerun" in json.loads(target.read_text())["mcpServers"]
    uninstall("claude-code", "user")
    assert link.is_symlink()
    assert "mobilerun" not in json.loads(target.read_text())["mcpServers"]
    assert json.loads(target.read_text())["other"] is True


def test_atomic_write_uses_portable_chmod(tmp_path, monkeypatch):
    from mobilerun.mcp import installer, utils

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        utils.os,
        "fchmod",
        lambda *args: pytest.fail("fchmod is not portable"),
        raising=False,
    )
    assert installer.install("cursor", "user") == "installed"


def test_custom_agent_uses_registry_format(tmp_path, monkeypatch):
    import json

    from mobilerun.mcp.installer import install

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "MOBILERUN_AGENTS",
        json.dumps(
            {
                "custom": {
                    "skills": None,
                    "mcp": ["custom.json", "custom.json"],
                    "mcp_format": {
                        "servers_key": "connections",
                        "entry_type": "http",
                        "api_key_ref": "${env:MOBILERUN_API_KEY}",
                    },
                }
            }
        ),
    )
    assert install("custom", "user", api_key=True) == "installed"
    entry = json.loads((tmp_path / "custom.json").read_text())["connections"][
        "mobilerun"
    ]
    assert entry["type"] == "http"
    assert entry["headers"]["Authorization"] == "Bearer ${env:MOBILERUN_API_KEY}"


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path / "opencode.jsonc"


@pytest.mark.parametrize("operation", [installer.install, installer.uninstall])
@pytest.mark.parametrize("repeated", [False, True])
def test_concurrent_editor_save(config, monkeypatch, operation, repeated):
    original = '{"mcp":{"servers":%s}}' % (
        "{}" if operation is installer.install else '{"mobilerun":{}}'
    )
    config.write_text(original)
    mkstemp = utils.tempfile.mkstemp
    saves = 0

    def editor_save(*args, **kwargs):
        nonlocal saves
        if repeated or not saves:
            saves += 1
            assert saves <= 10, "retry limit exceeded"
            config.write_text(original[:-1] + f',"editor":{saves}}}')
        return mkstemp(*args, **kwargs)

    monkeypatch.setattr(utils.tempfile, "mkstemp", editor_save)
    if repeated:
        with pytest.raises(ValueError, match="changed"):
            operation("opencode", "project")
        assert saves > 1
        assert config.read_text() == original[:-1] + f',"editor":{saves}}}'
    else:
        operation("opencode", "project")
        parsed = utils._loads(config.read_text())
        assert parsed["editor"] == 1
        assert ("mobilerun" in parsed["mcp"]["servers"]) == (
            operation is installer.install
        )
    assert not list(config.parent.glob(f".{config.name}.*"))


@pytest.mark.parametrize("comment", ["/* keep */", "// keep\n"])
@pytest.mark.parametrize(
    "left,right",
    [("", ',"a":{}'), ('"a":{},', ',"b":{}'), ('"a":{},', ","), ('"a":{},', "")],
)
def test_uninstall_preserves_adjacent_comments(config, comment, left, right):
    config.write_text(
        '{"mcp":{"servers":{'
        + left
        + '/* before */ "mobilerun":{} '
        + comment
        + right
        + "}}}"
    )
    assert installer.uninstall("opencode", "project") == "removed"
    text = config.read_text()
    assert "/* before */" in text and comment in text
    assert set(utils._loads(text)["mcp"]["servers"]) == {
        k for k in ("a", "b") if k in left + right
    }


@pytest.mark.parametrize("newline", [b"\r", b"\n", b"\r\n"])
def test_jsonc_line_comment_round_trip(config, newline):
    original = newline.join(
        [b'{"mcp":{"servers":{', b"  // keep", b'  "other":{}', b"}}}", b""]
    )
    config.write_bytes(original)
    assert installer.status("opencode", "project") == "not installed"
    assert installer.install("opencode", "project") == "installed"
    installer.install("opencode", "project", api_key=True, force=True)
    inserted = config.read_bytes().replace(newline, b"")
    assert b"\n" not in inserted and b"\r" not in inserted
    assert installer.uninstall("opencode", "project") == "removed"
    assert config.read_bytes() == original


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_jsonc_is_rejected(config, constant):
    text = '{"other":' + constant + "}"
    config.write_text(text)
    for operation in (installer.install, installer.status, installer.uninstall):
        with pytest.raises(ValueError, match=f"Invalid config .*{constant}"):
            operation("opencode", "project")
        assert config.read_text() == text


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
