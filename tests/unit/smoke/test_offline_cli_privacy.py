"""Exercise the installed CLI without devices, cloud, or real skill directories."""

import json
import subprocess
import sys


def test_offline_cli_protects_local_skill_and_macro_text(tmp_path):
    from mobilerun.skills.bundle import bundle_folders

    def cli(*args):
        return subprocess.run(
            [sys.executable, "-m", "mobilerun", *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )

    name = next(iter(bundle_folders()))
    target = tmp_path / "skills"
    options = ["--agent", "codex", "--dir", str(target)]
    installed = cli("skills", "install", "bundle", name, *options)
    assert installed.returncode == 0, installed.stderr
    notes = target / name / "user-notes.md"
    notes.write_text("keep me")
    refused = cli("skills", "uninstall", name, *options)
    assert refused.returncode != 0
    assert notes.read_text() == "keep me"

    macro = tmp_path / "macro.json"
    macro.write_text(
        json.dumps(
            {
                "macro_schema_version": "2.0",
                "actions": [
                    {
                        "type": "tap",
                        "x": 4,
                        "y": 5,
                        "pre_state": {"text": "SYNTHETIC-PRIVATE"},
                    }
                ],
            }
        )
    )
    preview = cli("macro", "replay", str(macro), "--dry-run")
    assert preview.returncode == 0, preview.stderr
    assert "tap x=4 y=5" in preview.stdout
    assert "SYNTHETIC-PRIVATE" not in preview.stdout + preview.stderr
