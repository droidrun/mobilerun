import pytest

from mobilerun.sandbox.errors import SandboxDeclarationError, SandboxPackagingError
from mobilerun.sandbox.packaging import transform

SOURCE = """from mobilerun.sandbox import cloud, Context
from pydantic import BaseModel
from datetime import datetime, timezone

class Item(BaseModel):
    at: datetime

class Input(BaseModel):
    items: list[Item]

UNUSED = 1

@cloud("example")
def main(input: Input, ctx: Context) -> Input | None:
    return input
"""


def test_extracts_nested_models_and_required_imports_only():
    result = transform(SOURCE)
    assert "class Item(" in result
    assert "class Input(" in result
    assert "from datetime import datetime" in result
    assert "timezone" not in result
    assert "UNUSED" not in result
    assert "from mobilerun_functions import " in result
    assert result.count("@script") == 1


def test_rejects_external_global_with_source_line():
    source = SOURCE.replace("return input", "return UNUSED")
    with pytest.raises(SandboxPackagingError, match=r"main.py:15:.*UNUSED"):
        transform(source)


def test_rejects_multiple_cloud_functions():
    source = SOURCE + '\n@cloud("other")\ndef other(): pass\n'
    with pytest.raises(SandboxDeclarationError):
        transform(source)


@pytest.mark.parametrize(
    "expression",
    [
        "os.path",
        "helper()",
        "UNUSED",
        "[json.dumps(i) for i in range(3)]",
        "(lambda: helper())()",
        "(lambda: UNUSED)()",
    ],
)
def test_global_reads_in_nested_scopes_are_rejected(expression):
    with pytest.raises(SandboxPackagingError, match="main.py:15:"):
        transform(SOURCE.replace("return input", "return " + expression))


def test_lazy_import_nested_closure_and_comprehension():
    source = SOURCE.replace(
        "    return input",
        '    import os\n    def inner(x):\n        return os.path.join(str(x), "a")\n    values = [inner(i) for i in range(3)]\n    return input',
    )
    assert "    import os" in transform(source)


def test_comprehension_shadow_does_not_hide_outer_global():
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(
            SOURCE.replace("return input", "return [UNUSED for UNUSED in []], UNUSED")
        )


def test_comprehension_target_shadows_global():
    transform(SOURCE.replace("return input", "return [UNUSED for UNUSED in []]"))


def test_global_statement_rejected():
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(
            SOURCE.replace("    return input", "    global UNUSED\n    return UNUSED")
        )


def test_all_global_read_lines_reported():
    source = SOURCE.replace(
        "    return input", "    print(UNUSED)\n    return helper()"
    )
    with pytest.raises(SandboxPackagingError) as error:
        transform(source)
    assert "main.py:15: 'UNUSED'" in str(error.value)
    assert "main.py:16: 'helper'" in str(error.value)


def test_shadow_reports_only_actual_global_read():
    source = SOURCE.replace(
        "    return input", "    values = [UNUSED for UNUSED in []]\n    return UNUSED"
    )
    with pytest.raises(SandboxPackagingError) as error:
        transform(source)
    assert str(error.value).count("'UNUSED'") == 1
    assert "main.py:16:" in str(error.value)


def test_nested_function_annotations_checked():
    source = SOURCE.replace(
        "    return input",
        "    def inner(x: UNUSED):\n        return x\n    return input",
    )
    with pytest.raises(SandboxPackagingError, match="'UNUSED'"):
        transform(source)


def test_future_annotations_forward_models():
    source = (
        SOURCE.replace(
            "from mobilerun.sandbox",
            "from __future__ import annotations\nfrom mobilerun.sandbox",
        )
        .replace("class Item(BaseModel):\n    at: datetime\n\n", "")
        .replace("UNUSED = 1", "class Item(BaseModel):\n    at: datetime\n\nUNUSED = 1")
        .replace("-> Input | None", '-> "Input | None"')
    )
    result = transform(source)
    assert "from __future__ import annotations" in result
    assert "class Item(" in result
    assert "class Input(" in result


def test_multiple_lambdas_on_same_line():
    with pytest.raises(SandboxPackagingError) as error:
        transform(
            SOURCE.replace("return input", "return (lambda: helper(), lambda: UNUSED)")
        )
    assert "'helper'" in str(error.value)
    assert "'UNUSED'" in str(error.value)


def test_dataset_nested_model():
    result = transform(
        SOURCE.replace("cloud, Context", "cloud, Context, Dataset").replace(
            "-> Input | None", "-> Dataset[Item]"
        )
    )
    assert "Dataset" in result
    assert "class Item(" in result


@pytest.mark.parametrize(
    "source",
    [
        SOURCE.replace(
            "from mobilerun.sandbox import cloud, Context",
            "from mobilerun.sandbox import Context",
        ),
        SOURCE.replace('@cloud("example")', "@script"),
        SOURCE.replace(
            "from mobilerun.sandbox import cloud, Context",
            "from mobilerun.sandbox import cloud\nfrom mobilerun.sandbox import cloud, Context",
        ),
        SOURCE.replace('@cloud("example")', '@cloud("example")\n@cloud("other")'),
        SOURCE.replace('@cloud("example")', '@other\n@cloud("example")'),
        SOURCE.replace("import cloud, Context", "import cloud as c, Context"),
    ],
)
def test_invalid_declarations_rejected(source):
    with pytest.raises(SandboxDeclarationError):
        transform(source)


@pytest.mark.parametrize(
    "change",
    [
        ("from datetime import datetime, timezone", "from .sibling import datetime"),
        ("    return input", "    from .sibling import thing\n    return thing"),
        (
            "def main(input: Input, ctx: Context)",
            "def main(input: Input, ctx: Context = UNUSED)",
        ),
        ("    at: datetime", "    at: datetime = UNUSED"),
    ],
)
def test_unsupported_dependencies_rejected(change):
    with pytest.raises(SandboxPackagingError):
        transform(SOURCE.replace(*change))


def test_external_constant_in_annotation_is_rejected():
    with pytest.raises(SandboxPackagingError, match="UNUSED"):
        transform(SOURCE.replace("input: Input", "input: UNUSED"))


def test_model_method_local_variables_are_not_global_dependencies():
    source = SOURCE.replace(
        "    at: datetime",
        "    at: datetime\n    def display(self):\n        value = str(self.at)\n        return value",
    )
    assert "def display" in transform(source)


def test_model_method_global_helper_is_rejected():
    source = SOURCE.replace(
        "    at: datetime",
        "    at: datetime\n    def display(self):\n        return helper(self.at)",
    )
    with pytest.raises(SandboxPackagingError, match="helper"):
        transform(source)


def test_sibling_lazy_import_rejected(tmp_path):
    (tmp_path / "sibling.py").write_text("value = 1\n")
    source = SOURCE.replace(
        "    return input", "    import sibling\n    return sibling.value"
    )
    with pytest.raises(SandboxPackagingError, match="local import"):
        transform(source, filename=str(tmp_path / "main.py"))


def test_nested_quoted_model_reference_is_extracted():
    source = SOURCE.replace("items: list[Item]", 'items: list["Item"]')
    assert "class Item(" in transform(source)


def test_shadowed_builtin_in_model_is_rejected():
    source = SOURCE.replace("UNUSED = 1", "str = helper()\nUNUSED = 1")
    source = source.replace("    at: datetime", "    at: str")
    with pytest.raises(SandboxPackagingError, match="'str'"):
        transform(source)


def test_model_quoted_external_reference_is_rejected():
    source = SOURCE.replace("    at: datetime", '    at: "MissingType"')
    with pytest.raises(SandboxPackagingError, match="MissingType"):
        transform(source)


def test_literal_string_values_are_not_forward_references():
    source = SOURCE.replace(
        "from pydantic import BaseModel",
        "from pydantic import BaseModel\nfrom typing import Literal",
    )
    source = source.replace("    at: datetime", '    at: Literal["ready", "waiting"]')
    result = transform(source)
    assert 'Literal["ready", "waiting"]' in result


def test_separate_runtime_alias_is_preserved():
    source = SOURCE.replace(
        "from pydantic import BaseModel",
        "from pydantic import BaseModel\nfrom mobilerun_functions import Dataset as Data",
    )
    source = source.replace("-> Input | None", "-> Data[Item]")
    assert "Dataset as Data" in transform(source)


def test_model_method_local_does_not_authorize_entry_global():
    source = """from mobilerun.sandbox import cloud
from pydantic import BaseModel
MISSING = 42
class Input(BaseModel):
    def display(self):
        MISSING = 1
        return MISSING
@cloud("example")
def main(input: Input):
    print(MISSING)
"""
    with pytest.raises(SandboxPackagingError, match="MISSING"):
        transform(source)


@pytest.mark.parametrize(
    "imports,annotation",
    [
        ("from typing import Literal as L", 'L["ready", "waiting"]'),
        ("import typing", 'typing.Literal["ready", "waiting"]'),
        ("from typing import Annotated as A", 'A[str, "metadata"]'),
        ("import typing as t", 't.Annotated[str, "metadata"]'),
    ],
)
def test_annotation_containers_resolve_import_bindings(imports, annotation):
    source = SOURCE.replace(
        "from pydantic import BaseModel", "from pydantic import BaseModel\n" + imports
    )
    source = source.replace("    at: datetime", "    at: " + annotation)
    assert annotation in transform(source)


@pytest.mark.parametrize(
    "prefix,decorator",
    [
        ("", "@cloud"),
        ("alias = cloud\n", '@alias("other")'),
        ("import mobilerun.sandbox as sandbox\n", '@sandbox.cloud("other")'),
        (
            "import mobilerun.sandbox as sandbox\nalias = sandbox.cloud\n",
            '@alias("other")',
        ),
    ],
)
def test_additional_unsupported_cloud_declaration_rejected(prefix, decorator):
    source = SOURCE + "\n" + prefix + decorator + "\ndef other(): pass\n"
    with pytest.raises(SandboxDeclarationError):
        transform(source)


def test_model_method_local_annotation_does_not_authorize_global():
    source = """from mobilerun.sandbox import cloud
from pydantic import BaseModel
MISSING = 42
class Input(BaseModel):
    def display(self):
        local: MISSING = 1
        return local
@cloud("example")
def main(input: Input):
    print(MISSING)
"""
    with pytest.raises(SandboxPackagingError, match="MISSING"):
        transform(source)
