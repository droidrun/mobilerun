"""YAML or JSON credentials with private, atomic writes."""

import os
import shlex
import stat
import tempfile
from contextlib import contextmanager

try:
    import fcntl
except ImportError:  # Advisory locks are unavailable on non-POSIX platforms.
    fcntl = None
from pathlib import Path

import yaml

from mobilerun.core.credentials.base import normalize
from mobilerun.core.credentials.dict import DictCredentials


class FileCredentials(DictCredentials):
    def __init__(self, path: str | Path, *, name: str | None = None) -> None:
        self.path = Path(path).expanduser()
        super().__init__(self._read(), name=name)

    def _read(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            return {}
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            if stat.S_IMODE(os.fstat(stream.fileno()).st_mode) & 0o077:
                raise PermissionError(
                    f"Credentials file is accessible to others. Run: chmod 600 {shlex.quote(str(self.path))}"
                )
            try:
                data = yaml.safe_load(stream) or {}
            except yaml.YAMLError:
                raise ValueError("Invalid credentials file format") from None
        if not isinstance(data, dict):
            raise ValueError("Credentials file must contain a mapping")
        normalize(data)
        return data

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _write(self, data):
        normalized = normalize(data)
        missing = []
        parent = self.path.parent
        while not parent.exists():
            missing.append(parent)
            parent = parent.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
        fd, path = tempfile.mkstemp(prefix=".credentials-", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                yaml.safe_dump(data, stream, sort_keys=False)
            os.replace(path, self.path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
        self._credentials = normalized

    async def save(
        self, package: str | None, name: str, fields: dict[str, str]
    ) -> None:
        with self._locked():
            data = self._read()
            if package is None and isinstance(data.get(name), str):
                data.setdefault(None, {})[name] = {"password": data.pop(name)}
            entries = data.setdefault(package, {})
            if not isinstance(entries, dict):
                raise ValueError("Package conflicts with a flat credential")
            entries.setdefault(name, {}).update(fields)
            self._write(data)

    async def update(
        self, package: str | None, name: str, field: str, value: str
    ) -> None:
        with self._locked():
            data = self._read()
            if package is None and isinstance(data.get(name), str):
                data.setdefault(None, {})[name] = {"password": data.pop(name)}
            if (
                package not in data
                or name not in data[package]
                or field not in data[package][name]
            ):
                raise ValueError("Credential field not found")
            data[package][name][field] = value
            self._write(data)

    async def delete(
        self, package: str | None, name: str, field: str | None = None
    ) -> None:
        with self._locked():
            data = self._read()
            if package is None and isinstance(data.get(name), str):
                data.setdefault(None, {})[name] = {"password": data.pop(name)}
            if package not in data or name not in data[package]:
                raise ValueError("Credential not found")
            if field is None:
                del data[package][name]
            else:
                if field not in data[package][name]:
                    raise ValueError("Credential field not found")
                del data[package][name][field]
            if not data[package]:
                del data[package]
            self._write(data)
