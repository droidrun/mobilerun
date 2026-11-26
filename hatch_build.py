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
        has_git = (root / ".git").exists()
        if has_git:
            tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
        elif manifest.is_file():
            # The sdist and Docker context supply the same tracked-file record.
            tracked = manifest.read_bytes()
        else:
            raise RuntimeError("Missing skills: run git submodule update --init")
        names = [Path(os.fsdecode(name)) for name in tracked.split(b"\0") if name]
        files = []
        for name in names:
            if name.is_absolute() or any(
                part.startswith(".") or part in {"tests", "__pycache__"}
                for part in name.parts
            ):
                continue
            path = root / name
            if path.resolve() != root.resolve() / name:
                continue
            if path.is_file():
                files.append(path)
            elif not has_git:
                raise RuntimeError(
                    f"SKILLS-RECORD lists a missing file: {name}. "
                    "Regenerate the record for Docker or rebuild the sdist."
                )
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
