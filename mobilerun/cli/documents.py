"""YAML 1.2 core scalars and JSON wire documents for routines."""

import io
import json
import math
import re

import click
import yaml


class DocumentLoader(yaml.SafeLoader):
    """Keep YAML 1.1 booleans, timestamps and sexagesimal numbers as strings."""

    yaml_implicit_resolvers = {
        key: [
            (tag, pattern)
            for tag, pattern in rules
            if tag
            not in {
                "tag:yaml.org,2002:bool",
                "tag:yaml.org,2002:int",
                "tag:yaml.org,2002:float",
                "tag:yaml.org,2002:timestamp",
                "tag:yaml.org,2002:merge",
            }
        ]
        for key, rules in yaml.SafeLoader.yaml_implicit_resolvers.items()
    }

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Document keys must be strings.")
            if key in result:
                raise ValueError(f"Duplicate document key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _integer(loader, node):
    value = loader.construct_scalar(node)
    sign = -1 if value.startswith("-") else 1
    value = value.lstrip("+-")
    if value.startswith("0o"):
        return sign * int(value[2:], 8)
    if value.startswith("0x"):
        return sign * int(value[2:], 16)
    return sign * int(value, 10)


DocumentLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
DocumentLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[+-]?(?:[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    list("-+0123456789"),
)
DocumentLoader.add_constructor("tag:yaml.org,2002:int", _integer)
DocumentLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^[+-]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+|\.(?:inf|Inf|INF|nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def _wire_value(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Document keys must be strings.")
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        "Document values must be JSON compatible (no NaN, infinity or YAML tags)."
    )


def read_document(file):
    try:
        text = file.read_text(encoding="utf-8")
        value = (
            json.loads(text)
            if file.suffix.lower() == ".json"
            else yaml.load(text, Loader=DocumentLoader)
        )
        value = _wire_value(value)
        if not isinstance(value, dict):
            raise ValueError("Expected a document object.")
        return value
    except (OSError, ValueError, yaml.YAMLError, RecursionError) as exc:
        raise click.ClickException(f"Cannot read {file} as JSON/YAML: {exc}") from None


def render_document(document, format):
    if "$schema" in document:
        document = {"$schema": document["$schema"], **document}
    if format == "json":
        return json.dumps(document, indent=2) + "\n"

    # Quote all strings so PyYAML's YAML 1.1 dumper cannot change 1.2 scalars
    # such as 1e3, 012, on and dates when this file is read again.
    class DocumentDumper(yaml.SafeDumper):
        pass

    DocumentDumper.add_representer(
        str,
        lambda dumper, value: dumper.represent_scalar(
            "tag:yaml.org,2002:str", value, style='"'
        ),
    )
    stream = io.StringIO()
    yaml.dump(
        document, stream, Dumper=DocumentDumper, sort_keys=False, allow_unicode=True
    )
    return stream.getvalue()
