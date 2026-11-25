"""Regression tests for concurrent and comment-preserving MCP config edits."""

import pytest

from mobilerun.mcp import installer, utils


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
