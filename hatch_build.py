"""Include the pinned skills tree in release archives."""

import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root) / "skills"
        manifest = Path(self.root) / "SKILLS-RECORD"
        if (root / ".git").exists():
            tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        elif manifest.is_file() and manifest.with_name("PKG-INFO").is_file():
            # Rebuild from the sdist without relying on Git or a directory scan.
            tracked = manifest.read_bytes()
        else:
            raise RuntimeError("Missing skills: run git submodule update --init")
        names = [Path(os.fsdecode(name)) for name in tracked.split(b"\0") if name]
        files = [
            root / name
            for name in names
            if not name.is_absolute()
            and not any(
                part.startswith(".") or part in {"tests", "__pycache__"}
                for part in name.parts
            )
            and (root / name).resolve() == root.resolve() / name
            and (root / name).is_file()
        ]
        if not any(path.name == "SKILL.md" for path in files):
            raise RuntimeError("Missing skills: run git submodule update --init")
        destination = (
            Path("mobilerun/assets/skills")
            if self.target_name == "wheel"
            else Path("skills")
        )
        build_data["force_include"].update(
            {str(path): str(destination / path.relative_to(root)) for path in files}
        )
        if self.target_name == "sdist":
            self._manifest_dir = TemporaryDirectory()
            record = Path(self._manifest_dir.name) / manifest.name
            record.write_bytes(
                b"\0".join(
                    os.fsencode(path.relative_to(root).as_posix()) for path in files
                )
            )
            build_data["force_include"][str(record)] = manifest.name

    def finalize(self, version, build_data, artifact_path):
        if self.target_name == "sdist":
            self._manifest_dir.cleanup()
