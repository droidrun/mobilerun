from mobilerun.tools.filters import ConciseFilter, DetailedFilter
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.helpers.element_search import flatten_tree


def test_null_children_are_treated_as_empty():
    tree = {
        "text": "root",
        "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 100},
        "children": [
            {
                "text": "leaf",
                "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 100},
                "children": None,
            }
        ],
    }
    context = {"screen_bounds": {"width": 100, "height": 100}}

    assert ConciseFilter().filter(tree, context)["children"][0]["children"] == []
    assert DetailedFilter().filter(tree, context)["children"][0]["children"] == []
    assert [node["text"] for node in flatten_tree(tree)] == ["root", "leaf"]

    _, _, elements, _ = IndexedFormatter().format(tree, {})
    assert [element["text"] for element in elements] == ["root", "leaf"]
