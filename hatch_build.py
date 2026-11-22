"""Include the pinned skills tree in release archives."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        root = Path(self.root) / "skills"
        files = [
            path
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and not any(
                part.startswith(".git") or part in {"tests", "__pycache__"}
                for part in path.relative_to(root).parts
            )
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
