"""Extract one cloud declaration without executing its containing module.

Inspection imports the generated script and its required dependencies in a
subprocess. Only inspect trusted source and trusted Python environments.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import subprocess
import symtable
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import SandboxDeclarationError, SandboxPackagingError
from .models import Artifact, CloudDeclaration
from .settings import RUNTIME_MODULE, get_sandbox_settings

JSON = dict[str, Any]


@dataclass(frozen=True)
class _ScheduledArtifact(Artifact):
    # Keep evidence tied to the exact source even across dataclasses.replace().
    inspected_source: str


def verify_scheduled_artifact(artifact: Artifact) -> None:
    """Inspect a supplied schedule unless its source already passed the worker check."""
    if (
        not isinstance(artifact, _ScheduledArtifact)
        or artifact.inspected_source != artifact.source
    ):
        inspect_source(artifact.source, scheduled=True)


def _parse(source: str, filename: str) -> ast.Module:
    try:
        return ast.parse(source, filename=filename)
    except (SyntaxError, ValueError) as exc:
        raise SandboxDeclarationError(f"{filename}: invalid Python source") from exc


def _declaration(
    tree: ast.Module, filename: str
) -> tuple[CloudDeclaration, ast.FunctionDef | ast.AsyncFunctionDef]:
    imports = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        and n.module == "mobilerun.sandbox"
        and n.level == 0
        and any(a.name == "cloud" or a.name == "*" for a in n.names)
    ]
    candidates = [
        (n, d)
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        for d in n.decorator_list
        if isinstance(d, ast.Call)
        and isinstance(d.func, ast.Name)
        and d.func.id == "cloud"
    ]
    if len(imports) != 1 or imports[0] not in tree.body or len(candidates) != 1:
        raise SandboxDeclarationError(
            f"{filename}: expected one canonical cloud import and one @cloud(...) function"
        )
    imp = imports[0]
    if (
        any(a.name == "*" or a.asname for a in imp.names)
        or sum(a.name == "cloud" for a in imp.names) != 1
    ):
        raise SandboxDeclarationError(
            f"{filename}:{imp.lineno}: aliases and star imports are not supported"
        )
    cloud_names = {"cloud"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "mobilerun.sandbox":
            cloud_names.update(
                a.asname or a.name for a in node.names if a.name == "cloud"
            )
        if isinstance(node, ast.Assign) and (
            (isinstance(node.value, ast.Name) and node.value.id in cloud_names)
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "cloud")
        ):
            cloud_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            is_cloud = (isinstance(target, ast.Name) and target.id in cloud_names) or (
                isinstance(target, ast.Attribute) and target.attr == "cloud"
            )
            if is_cloud and (node, dec) not in candidates:
                raise SandboxDeclarationError(
                    f"{filename}:{dec.lineno}: unsupported additional cloud declaration"
                )
    func, decorator = candidates[0]
    if len(func.decorator_list) != 1:
        raise SandboxDeclarationError(
            f"{filename}:{func.lineno}: stacked decorators are not supported"
        )
    if len(decorator.args) > 1 or any(k.arg is None for k in decorator.keywords):
        raise SandboxDeclarationError(
            f"{filename}:{decorator.lineno}: invalid cloud options"
        )
    values: dict[str, Any] = {}
    try:
        if decorator.args:
            values["name"] = ast.literal_eval(decorator.args[0])
        for keyword in decorator.keywords:
            if keyword.arg == "trigger":
                raise SandboxDeclarationError(
                    "The manual trigger was removed; use an endpoint with device= to attach a device"
                )
            if (
                keyword.arg not in {"name", "schedule", "device", "endpoint"}
                or keyword.arg in values
            ):
                raise ValueError("unknown or duplicate option")
            values[keyword.arg] = ast.literal_eval(keyword.value)
        declaration = CloudDeclaration(**values)
        declaration.validate()
    except SandboxDeclarationError as exc:
        raise SandboxDeclarationError(f"{filename}:{decorator.lineno}: {exc}") from exc
    except (TypeError, ValueError, SyntaxError) as exc:
        raise SandboxDeclarationError(
            f"{filename}:{decorator.lineno}: invalid literal cloud options"
        ) from exc
    return declaration, func


def read_declaration(path: Path) -> CloudDeclaration:
    """Read a single literal declaration without importing the user's file."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SandboxDeclarationError(f"Cannot read declaration: {path}") from exc
    return _declaration(_parse(source, str(path)), str(path))[0]


def _names(node: ast.AST, bindings: dict[str, str]) -> set[str]:
    result = {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    for n in ast.walk(node):
        if isinstance(n, ast.AnnAssign):
            result.update(_annotation_names(n.annotation, bindings))
    return result


def _annotation_names(node: ast.AST, bindings: dict[str, str]) -> set[str]:
    result: set[str] = set()

    class AnnotationVisitor(ast.NodeVisitor):
        def visit_Name(self, part: ast.Name) -> None:
            if isinstance(part.ctx, ast.Load):
                result.add(part.id)

        def visit_Constant(self, part: ast.Constant) -> None:
            if not isinstance(part.value, str):
                return
            try:
                self.visit(ast.parse(part.value, mode="eval"))
            except SyntaxError:
                pass

        def visit_Subscript(self, part: ast.Subscript) -> None:
            self.visit(part.value)
            if isinstance(part.value, ast.Name):
                name = bindings.get(part.value.id, "")
            elif (
                isinstance(part.value, ast.Attribute)
                and isinstance(part.value.value, ast.Name)
                and bindings.get(part.value.value.id) in ("typing", "typing_extensions")
            ):
                name = part.value.attr
            else:
                name = ""
            if name == "Literal":
                return
            if name == "Annotated" and isinstance(part.slice, ast.Tuple):
                self.visit(part.slice.elts[0])
                for item in part.slice.elts[1:]:
                    result.update(_names(item, bindings))
                return
            self.visit(part.slice)

    AnnotationVisitor().visit(node)
    return result


def _local_import(node: ast.Import | ast.ImportFrom, filename: str) -> bool:
    if isinstance(node, ast.ImportFrom) and node.level:
        return True
    modules = (
        [node.module or ""]
        if isinstance(node, ast.ImportFrom)
        else [a.name for a in node.names]
    )
    root = Path(filename).absolute().parent
    return any(
        (root / (name.split(".")[0] + ".py")).exists()
        or (root / name.split(".")[0] / "__init__.py").exists()
        for name in modules
        if name
    )


class _GlobalReads(ast.NodeVisitor):
    def __init__(
        self,
        scope: symtable.SymbolTable,
        filename: str,
        allowed: set[str],
        bound: set[str],
        violations: list[tuple[int, str]],
        used: set[int],
    ):
        self.scope, self.filename, self.allowed, self.bound = (
            scope,
            filename,
            allowed,
            bound,
        )
        self.violations, self.used, self.hidden = violations, used, set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            try:
                is_global = self.scope.lookup(node.id).is_global()
            except KeyError:
                # Deferred annotations are absent from Python's symbol table.
                is_global = True
            if (
                is_global
                and node.id not in self.hidden
                and node.id not in self.allowed
                and (node.id not in dir(builtins) or node.id in self.bound)
            ):
                self.violations.append((node.lineno, node.id))

    def visit_Import(self, node: ast.Import) -> None:
        if _local_import(node, self.filename):
            raise SandboxPackagingError(
                f"{self.filename}:{node.lineno}: local imports are not supported"
            )

    visit_ImportFrom = visit_Import

    def child(self, node: ast.AST, name: str) -> _GlobalReads:
        table = next(
            (
                t
                for t in self.scope.get_children()
                if t.get_name() == name
                and t.get_lineno() == node.lineno
                and t.get_id() not in self.used
            ),
            None,
        )
        if table is not None:
            self.used.add(table.get_id())
        visitor = _GlobalReads(
            table or self.scope,
            self.filename,
            self.allowed,
            self.bound,
            self.violations,
            self.used,
        )
        visitor.hidden = set(self.hidden)
        return visitor

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        args = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
            node.args.vararg,
            node.args.kwarg,
        ]
        parts = [
            *node.decorator_list,
            *node.args.defaults,
            *[d for d in node.args.kw_defaults if d],
            *[a.annotation for a in args if a and a.annotation],
        ]
        if node.returns:
            parts.append(node.returns)
        for part in parts:
            self.visit(part)
        visitor = self.child(node, node.name)
        for part in node.body:
            visitor.visit(part)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node: ast.Lambda) -> None:
        for part in [*node.args.defaults, *[d for d in node.args.kw_defaults if d]]:
            self.visit(part)
        self.child(node, "lambda").visit(node.body)

    def _comprehension(self, node: Any, name: str, output: list[ast.AST]) -> None:
        self.visit(node.generators[0].iter)
        visitor = self.child(node, name)
        for index, generator in enumerate(node.generators):
            if index:
                visitor.visit(generator.iter)
            visitor.hidden.update(
                n.id
                for n in ast.walk(generator.target)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            )
            for condition in generator.ifs:
                visitor.visit(condition)
        for part in output:
            visitor.visit(part)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._comprehension(node, "listcomp", [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._comprehension(node, "setcomp", [node.elt])

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._comprehension(node, "genexpr", [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._comprehension(node, "dictcomp", [node.key, node.value])

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for part in [
            *node.decorator_list,
            *node.bases,
            *(k.value for k in node.keywords),
        ]:
            self.visit(part)
        visitor = self.child(node, node.name)
        for part in node.body:
            visitor.visit(part)


def transform(source: str, *, filename: str = "main.py") -> str:
    """Extract a standalone script while rejecting omitted dependencies."""
    tree = _parse(source, filename)
    _, func = _declaration(tree, filename)
    args = [
        *func.args.posonlyargs,
        *func.args.args,
        *func.args.kwonlyargs,
        func.args.vararg,
        func.args.kwarg,
    ]
    annotations = [a.annotation for a in args if a and a.annotation]
    if func.returns:
        annotations.append(func.returns)
    imports: dict[str, ast.Import | ast.ImportFrom] = {}
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.asname or (
                    alias.name.split(".")[0]
                    if isinstance(node, ast.Import)
                    else alias.name
                )
                imports[name] = node
                if isinstance(node, ast.Import) and alias.name in (
                    "typing",
                    "typing_extensions",
                ):
                    bindings[name] = alias.name
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module in ("typing", "typing_extensions")
                    and alias.name in ("Literal", "Annotated")
                ):
                    bindings[name] = alias.name
    table = symtable.symtable(source, filename, "exec")
    function_table = next(
        t
        for t in table.get_children()
        if t.get_name() == func.name and t.get_lineno() == func.lineno
    )
    bound = {
        s.get_name()
        for s in table.get_symbols()
        if s.is_assigned() or s.is_imported() or s.is_namespace()
    }
    required = (
        set().union(*(_annotation_names(n, bindings) for n in annotations))
        if annotations
        else set()
    )
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    selected: set[str] = set()
    while pending := (required & classes.keys()) - selected:
        for name in pending:
            selected.add(name)
            reads: list[tuple[int, str]] = []
            _GlobalReads(table, filename, set(), bound, reads, set()).visit(
                classes[name]
            )
            required.update(dependency for _, dependency in reads)
            for node in ast.walk(classes[name]):
                if isinstance(node, ast.AnnAssign):
                    required.update(_annotation_names(node.annotation, bindings))
    allowed = selected | (required & imports.keys()) | {"Context"}
    violations: list[tuple[int, str]] = []
    for annotation in annotations:
        for name in _annotation_names(annotation, bindings):
            if (
                name not in classes
                and name not in imports
                and (name not in dir(builtins) or name in bound)
            ):
                violations.append((annotation.lineno, name))
    visitor = _GlobalReads(function_table, filename, allowed, bound, violations, set())
    for statement in func.body:
        visitor.visit(statement)
    for default in [*func.args.defaults, *[d for d in func.args.kw_defaults if d]]:
        for node in ast.walk(default):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and (node.id not in dir(builtins) or node.id in bound)
            ):
                violations.append((node.lineno, node.id))
    for name in selected:
        model_visitor = _GlobalReads(
            table, filename, set(imports) | selected, bound, violations, set()
        )
        model_visitor.visit(classes[name])
        for node in ast.walk(classes[name]):
            if isinstance(node, ast.AnnAssign):
                for dependency in _annotation_names(node.annotation, bindings):
                    if (
                        dependency not in imports
                        and dependency not in selected
                        and (dependency not in dir(builtins) or dependency in bound)
                    ):
                        violations.append((node.lineno, dependency))
    if violations:
        raise SandboxPackagingError(
            "\n".join(
                f"{filename}:{line}: '{name}' is defined outside {func.name} — import or define it inside the function (lazy import)"
                for line, name in sorted(set(violations))
            )
        )
    future = any(
        isinstance(n, ast.ImportFrom)
        and n.module == "__future__"
        and any(a.name == "annotations" for a in n.names)
        for n in tree.body
    )
    generated = ["from __future__ import annotations"] if future else []
    runtime_names = ["Context", "script"]
    for name in sorted(required):
        node = imports.get(name)
        if (
            isinstance(node, ast.ImportFrom)
            and node.module in ("mobilerun.sandbox", RUNTIME_MODULE)
            and name not in ("cloud", "Context", "script")
        ):
            alias = next(a for a in node.names if (a.asname or a.name) == name)
            runtime_names.append(
                alias.name + (" as " + alias.asname if alias.asname else "")
            )
    generated.append(f"from {RUNTIME_MODULE} import " + ", ".join(runtime_names))
    for node in tree.body:
        if (
            not isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.ImportFrom)
            and node.module in ("mobilerun.sandbox", RUNTIME_MODULE, "__future__")
        ):
            continue
        aliases = [
            a
            for a in node.names
            if (
                a.asname
                or (a.name.split(".")[0] if isinstance(node, ast.Import) else a.name)
            )
            in required
        ]
        if not aliases:
            continue
        if _local_import(node, filename):
            raise SandboxPackagingError(
                f"{filename}:{node.lineno}: local imports are not supported"
            )
        text = ", ".join(
            a.name + (" as " + a.asname if a.asname else "") for a in aliases
        )
        generated.append(
            (
                "import "
                if isinstance(node, ast.Import)
                else f"from {node.module} import "
            )
            + text
        )
    lines = source.splitlines(keepends=True)
    definitions = [
        "".join(
            lines[
                min([n.lineno, *(d.lineno for d in n.decorator_list)])
                - 1 : n.end_lineno
            ]
        ).rstrip()
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name in selected
    ]
    body = "".join(lines[func.lineno - 1 : func.end_lineno]).rstrip()
    return (
        "\n".join(generated)
        + "\n\n\n"
        + "\n\n\n".join([*definitions, "@script\n" + body])
        + "\n"
    )


# Use the same loader and inspector as the runtime's `inspect` command. Run in
# its existing child process: never import the user's original module or call
# the function. Only scheduled inspections need the additional model check.
_SCHEDULE_INSPECTOR = f"""\
import json, sys
from contextlib import redirect_stdout
from pydantic import ValidationError
from {RUNTIME_MODULE}.loader import load_script
from {RUNTIME_MODULE}._contract import inspect_entry

with redirect_stdout(sys.stderr):
    entry = load_script(sys.argv[1])
    document = inspect_entry(entry)
    try:
        entry.input_model.model_validate_json('{{}}')
    except ValidationError as exc:
        properties = document['inputSchema'].get('properties', {{}})
        locations = (error['loc'] for error in exc.errors(
            include_input=False, include_context=False, include_url=False))
        document['scheduleInputErrors'] = sorted({{
            loc[0] for loc in locations if loc and loc[0] in properties
        }})
    else:
        document['scheduleInputErrors'] = None
print(json.dumps(document))
"""


def inspect_source(source: str, *, scheduled: bool = False) -> JSON:
    """Inspect trusted generated source in an isolated temporary directory."""
    try:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script.py"
            path.write_text(source, encoding="utf-8")
            args = (
                ["-c", _SCHEDULE_INSPECTOR]
                if scheduled
                else ["-m", RUNTIME_MODULE, "inspect"]
            )
            result = subprocess.run(
                [sys.executable, *args, str(path)],
                capture_output=True,
                text=True,
                timeout=get_sandbox_settings().MOBILERUN_SANDBOX_INSPECTION_TIMEOUT,
                check=False,
            )
        if result.returncode:
            raise SandboxPackagingError("Local sandbox inspection failed")
        document = json.loads(result.stdout)
        if (
            not isinstance(document, dict)
            or document.get("ok") is not True
            or not isinstance(document.get("inputSchema"), dict)
            or not (
                document.get("outputSchema") is None
                or isinstance(document.get("outputSchema"), dict)
            )
            or "outputSchema" not in document
            or (scheduled and "scheduleInputErrors" not in document)
        ):
            raise SandboxPackagingError(
                "Local sandbox inspection returned an invalid document"
            )
        if scheduled and (fields := document["scheduleInputErrors"]) is not None:
            properties = document["inputSchema"].get("properties", {})
            if not isinstance(fields, list) or not all(
                isinstance(field, str) and field in properties for field in fields
            ):
                raise SandboxPackagingError("Invalid scheduled input inspection result")
            raise SandboxDeclarationError(
                "Scheduled input must accept {}"
                + (f"; failing fields: {', '.join(fields)}" if fields else "")
            )
        return document
    except (
        OSError,
        UnicodeError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ):
        raise SandboxPackagingError("Local sandbox inspection failed") from None


def package(path: Path) -> Artifact:
    """Build and inspect an artifact before any cloud operation."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise SandboxPackagingError(f"Cannot read source: {path}") from exc
    declaration = _declaration(_parse(source, str(path)), str(path))[0]
    generated = transform(source, filename=str(path))
    scheduled = declaration.schedule is not None
    document = inspect_source(generated, scheduled=scheduled)
    digest = hashlib.sha256(
        RUNTIME_MODULE.encode("utf-8") + b"\0" + generated.encode("utf-8")
    ).hexdigest()
    if scheduled:
        return _ScheduledArtifact(
            declaration,
            generated,
            digest,
            document["inputSchema"],
            document["outputSchema"],
            inspected_source=generated,
        )
    return Artifact(
        declaration,
        generated,
        digest,
        document["inputSchema"],
        document["outputSchema"],
    )
