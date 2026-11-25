# skills/

Discover, install, and remove bundled coding-agent skills.

**What belongs here / what doesn't:** Bundled skill management belongs here; agent runtime code does not.

## Contents

| Path | Purpose |
| --- | --- |
| `__init__.py` | Public exports |
| `bundle.py` | Skill discovery, installation, and removal |
| `utils.py` | Metadata parsing and safe directory installation helpers |
| [cli/](cli/README.md) | Skill management commands |

## Quick example

```python
from mobilerun.skills import list_skills

skills = list_skills()
print([skill.name for skill in skills])
```

```sh
mobilerun skills install mobilerun-cloud --agent claude-code
```

Installation writes to the selected agent's skills directory. See
[skill CLI commands](cli/README.md).

`mobilerun skills show NAME` reports unknown names and points to
`mobilerun skills list`. When a bundled name is close, it also suggests a match.
Installing an unknown bundled skill reports the same guidance.

**Depends on / used by:** Uses the shared base. Independent of macro/, sandbox/, and agents/; only shared CLI helper `cli/base/` is allowed.

Skills are discovered recursively from the `skills` submodule. Names come from
frontmatter, or the directory when absent. Duplicate names are errors. The names
`bundle` and `single` are reserved for install modes.
Named `GUIDE.md` files inside a skill are discoverable sub-skills; their
`parent` identifies the containing skill. Installing one individually also
installs its parent if missing and adds a pointer to the parent for relative
paths. Uninstalling the sub-skill leaves its parent intact.
Markdown files directly under a skill's `references/` directory are also
discoverable sub-skills. Their names default to the parent name and file stem.

`mobilerun skills install --all --agent codex` copies all top-level folders
containing skills. `install bundle FOLDER --agent codex` copies one whole
folder; `install single NAME --agent codex` copies one skill flat. Existing
`install NAME` remains supported. Both modes support `--all`.

Forced reinstall and uninstall check installed files against their recorded hashes.
If a skill contains added or modified files, the CLI lists up to ten changed paths
and asks before discarding them. No is the default. In scripts, uninstall requires
`--discard-changes` to remove local edits; reinstall requires both `--force` and
`--discard-changes` to replace them. `--yes` alone never discards edits. Unmanaged
skills are not replaced or removed, even with `--discard-changes`.

Concurrent installs into the same skills directory are not supported.
