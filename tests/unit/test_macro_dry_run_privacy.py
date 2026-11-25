import json

from click.testing import CliRunner

from mobilerun.macro.cli import macro


def test_preview_shows_coordinates_without_snapshot_or_payload(tmp_path):
    secret = "PRIVATE-UI-TEXT" * 1000
    actions = [
        {
            "action_type": "tap",
            "x": 10,
            "y": 20,
            "pre_state": {"text": secret},
            "post_state": {"text": secret},
        },
        {"type": "input_text", "text": secret},
        {"type": "extension_call", "arguments": {"password": secret}},
    ]
    path = tmp_path / "macro.json"
    path.write_text(json.dumps({"macro_schema_version": "2.0", "actions": actions}))
    result = CliRunner().invoke(macro, ["replay", str(path), "--dry-run"])
    assert result.exit_code == 0, result.exception
    assert "PRIVATE-UI-TEXT" not in result.output
    assert "1: tap x=10 y=20" in result.output
    assert "input_text [redacted]" in result.output
    assert "extension_call [redacted]" in result.output
    assert len(result.output) < 200
