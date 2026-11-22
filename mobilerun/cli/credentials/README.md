# Credentials

Cloud agent tasks use the vault. Log in with `mobilerun account login`.
Local credentials are stored in a private YAML file and used by local devices;
they are not sent to cloud tasks. `MOBILERUN_CREDENTIALS_FILE` overrides the
local path. New files and their newly created parent directories are private;
existing files must have no group or world permissions (`chmod 600 FILE`).

## Quick example

```sh
mobilerun credentials list                    # vault and local metadata
mobilerun credentials list --vault --json
mobilerun credentials list --local
mobilerun credentials path
mobilerun credentials save                    # interactive store, app and fields
mobilerun credentials save com.example.app default --field username --field password
mobilerun credentials save com.example.app default --local --from-json credentials.json
cat credentials.json | mobilerun credentials save com.example.app default --from-json -
mobilerun credentials update com.example.app default password --local
mobilerun credentials update com.example.app default password --stdin < password.txt
mobilerun credentials delete com.example.app default password --local
mobilerun credentials delete com.example.app default --yes
```

Only package, name, field names and source are listed. JSON input is an object
mapping supported field names to string values. Prompts hide input. Save confirms
before modifying an existing credential; deletion also requires confirmation
unless `--yes` is supplied. Interactive update and delete can select missing
identifiers. Protect any secret input files and remove them when no longer needed.

Cloud stores dotted package names as dashed identifiers. Some existing dashed
names cannot be reversed unambiguously, so they may be shown as stored. Dotted
names are accepted for lookups.

## Files

| Path | Purpose |
| --- | --- |
| `group.py` | Credential commands and CLI argument handling |
| `__init__.py` | Package marker |

See the [CLI overview](../README.md) and [configuration](../../config/README.md).
