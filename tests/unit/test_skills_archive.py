"""Verify both release archives and discovery from an installed wheel."""

import json
import os
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path
from zipfile import ZipFile

import pytest


def test_skills_wheel_and_sdist_include_full_tree_and_installed_cli(tmp_path):
    pytest.importorskip("hatchling")
    root = Path(__file__).resolve().parents[2]
    source = root / "skills"
    assert (source / "mobile-harness/SKILL.md").is_file()
    files = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
        and not any(
            part.startswith(".git") or part in {"tests", "__pycache__"}
            for part in path.relative_to(source).parts
        )
    }
    assert "apps/com.ebay.mobile/scrape_newest_listings.md" in files
    dist = tmp_path / "dist"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "hatchling",
            "build",
            "-t",
            "wheel",
            "-t",
            "sdist",
            "-d",
            str(dist),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel = next(dist.glob("*.whl"))
    with ZipFile(wheel) as archive:
        actual = {
            name.removeprefix("mobilerun/assets/skills/"): archive.read(name)
            for name in archive.namelist()
            if name.startswith("mobilerun/assets/skills/")
        }
        assert actual == files
    with tarfile.open(next(dist.glob("*.tar.gz"))) as archive:
        actual = {}
        for member in archive.getmembers():
            if member.isfile() and "/skills/" in member.name:
                prefix, relative = member.name.split("/skills/", 1)
                if "/" not in prefix:
                    actual[relative] = archive.extractfile(member).read()
        assert actual == files

    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    python = venv / "bin/python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
        text=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = sysconfig.get_paths()["purelib"]
    command = "import sys,sysconfig; sys.path.insert(0,sysconfig.get_paths()['purelib']); from mobilerun.cli.app import cli; cli()"

    def run(*args):
        return subprocess.run(
            [str(python), "-c", command, *args],
            cwd=tmp_path,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    entries = json.loads(run("skills", "list", "--json").stdout)
    assert {entry["source"] for entry in entries} == {"skills"}
    assert {entry["name"] for entry in entries} >= {"mobile-harness", "mobilerun-cloud"}
    target = tmp_path / "target"
    run(
        "skills",
        "install",
        "single",
        "mobilerun-cloud",
        "--agent",
        "codex",
        "--dir",
        str(target),
    )
    assert (target / "mobilerun-cloud/SKILL.md").is_file()
