"""
Element search and filtering for Android accessibility trees.

Provides composable filters to search elements by text, ID, spatial relationships, and traits.
Works with raw a11y tree data from Portal before index assignment.
"""

from typing import List, Dict, Any, Callable, Tuple
import re

ElementFilter = Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]


# ========== HELPER FUNCTIONS ==========


def flatten_tree(root: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively flatten tree to list of all nodes."""
    results = [root]
    for child in root.get("children", []):
        results.extend(flatten_tree(child))
    return results


def get_element_center(node: Dict[str, Any]) -> Tuple[int, int]:
    """Get center coordinates from boundsInScreen."""
    bounds = node.get("boundsInScreen", {})
    left = bounds.get("left", 0)
    top = bounds.get("top", 0)
    right = bounds.get("right", 0)
    bottom = bounds.get("bottom", 0)

    center_x = (left + right) // 2
    center_y = (top + bottom) // 2

    return (center_x, center_y)


def sort_by_position(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort elements top-to-bottom, left-to-right."""

    def get_sort_key(node: Dict) -> Tuple[int, int]:
        bounds = node.get("boundsInScreen", {})
        top = bounds.get("top", 0)
        left = bounds.get("left", 0)
        return (top, left)

    return sorted(nodes, key=get_sort_key)


# ========== FILTERS CLASS ==========


class Filters:
    """Composable filters for element search."""

    # ========== TEXT MATCHING ==========

    @staticmethod
    def text_matches(pattern: str | re.Pattern) -> ElementFilter:
        """
        Match elements by text content in text, contentDescription, or hint fields.
        Supports exact match, regex, and newline normalization.
        """
        if isinstance(pattern, str):
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
            pattern_str = pattern
        else:
            regex = pattern
            pattern_str = pattern.pattern

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            results = []

            for node in all_nodes:
                text = node.get("text", "")
                content_desc = node.get("contentDescription", "")
                hint = node.get("hint", "")

                for field_value in [text, content_desc, hint]:
                    if not field_value:
                        continue

                    # Exact match
                    if pattern_str == field_value:
                        results.append(node)
                        break

                    # Regex match
                    if regex.search(field_value):
                        results.append(node)
                        break

                    # Newline-normalized match
                    normalized = field_value.replace("\n", " ")
                    if pattern_str == normalized or regex.search(normalized):
                        results.append(node)
                        break

            return results

        return filter_fn

    @staticmethod
    def id_matches(pattern: str | re.Pattern) -> ElementFilter:
        """Match elements by resource ID (full or short form)."""
        if isinstance(pattern, str):
            regex = re.compile(re.escape(pattern), re.IGNORECASE)
            pattern_str = pattern
        else:
            regex = pattern
            pattern_str = pattern.pattern

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            results = []

            for node in all_nodes:
                resource_id = node.get("resourceId", "")

                if not resource_id:
                    continue

                short_id = (
                    resource_id.split("/")[-1] if "/" in resource_id else resource_id
                )

                # Check full ID
                if pattern_str == resource_id or regex.search(resource_id):
                    results.append(node)
                    continue

                # Check short ID
                if pattern_str == short_id or regex.search(short_id):
                    results.append(node)

            return results

        return filter_fn

    # ========== SPATIAL FILTERS ==========

    @staticmethod
    def below(anchor_filter: ElementFilter) -> ElementFilter:
        """Find elements positioned below the anchor element."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            anchor_results = anchor_filter(nodes)

            if not anchor_results:
                return []

            anchor = anchor_results[0]
            anchor_bounds = anchor.get("boundsInScreen", {})
            anchor_bottom = anchor_bounds.get("bottom", 0)
            anchor_center_x, anchor_center_y = get_element_center(anchor)

            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            candidates = []
            for node in all_nodes:
                if node == anchor:
                    continue

                node_bounds = node.get("boundsInScreen", {})
                node_top = node_bounds.get("top", 0)

                if node_top > anchor_bottom:
                    node_center_x, node_center_y = get_element_center(node)
                    distance = (
                        (node_center_x - anchor_center_x) ** 2
                        + (node_center_y - anchor_center_y) ** 2
                    ) ** 0.5
                    candidates.append((distance, node))

            candidates.sort(key=lambda x: x[0])
            return [node for dist, node in candidates]

        return filter_fn

    @staticmethod
    def above(anchor_filter: ElementFilter) -> ElementFilter:
        """Find elements positioned above the anchor element."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            anchor_results = anchor_filter(nodes)

            if not anchor_results:
                return []

            anchor = anchor_results[0]
            anchor_bounds = anchor.get("boundsInScreen", {})
            anchor_top = anchor_bounds.get("top", 0)
            anchor_center_x, anchor_center_y = get_element_center(anchor)

            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            candidates = []
            for node in all_nodes:
                if node == anchor:
                    continue

                node_bounds = node.get("boundsInScreen", {})
                node_bottom = node_bounds.get("bottom", 0)

                if node_bottom < anchor_top:
                    node_center_x, node_center_y = get_element_center(node)
                    distance = (
                        (node_center_x - anchor_center_x) ** 2
                        + (node_center_y - anchor_center_y) ** 2
                    ) ** 0.5
                    candidates.append((distance, node))

            candidates.sort(key=lambda x: x[0])
            return [node for dist, node in candidates]

        return filter_fn

    @staticmethod
    def left_of(anchor_filter: ElementFilter) -> ElementFilter:
        """Find elements positioned left of the anchor element."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            anchor_results = anchor_filter(nodes)

            if not anchor_results:
                return []

            anchor = anchor_results[0]
            anchor_bounds = anchor.get("boundsInScreen", {})
            anchor_left = anchor_bounds.get("left", 0)
            anchor_center_x, anchor_center_y = get_element_center(anchor)

            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            candidates = []
            for node in all_nodes:
                if node == anchor:
                    continue

                node_bounds = node.get("boundsInScreen", {})
                node_right = node_bounds.get("right", 0)

                if node_right < anchor_left:
                    node_center_x, node_center_y = get_element_center(node)
                    distance = (
                        (node_center_x - anchor_center_x) ** 2
                        + (node_center_y - anchor_center_y) ** 2
                    ) ** 0.5
                    candidates.append((distance, node))

            candidates.sort(key=lambda x: x[0])
            return [node for dist, node in candidates]

        return filter_fn

    @staticmethod
    def right_of(anchor_filter: ElementFilter) -> ElementFilter:
        """Find elements positioned right of the anchor element."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            anchor_results = anchor_filter(nodes)

            if not anchor_results:
                return []

            anchor = anchor_results[0]
            anchor_bounds = anchor.get("boundsInScreen", {})
            anchor_right = anchor_bounds.get("right", 0)
            anchor_center_x, anchor_center_y = get_element_center(anchor)

            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            candidates = []
            for node in all_nodes:
                if node == anchor:
                    continue

                node_bounds = node.get("boundsInScreen", {})
                node_left = node_bounds.get("left", 0)

                if node_left > anchor_right:
                    node_center_x, node_center_y = get_element_center(node)
                    distance = (
                        (node_center_x - anchor_center_x) ** 2
                        + (node_center_y - anchor_center_y) ** 2
                    ) ** 0.5
                    candidates.append((distance, node))

            candidates.sort(key=lambda x: x[0])
            return [node for dist, node in candidates]

        return filter_fn

    # ========== TRAIT FILTERS ==========

    @staticmethod
    def clickable() -> ElementFilter:
        """Match clickable elements."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [node for node in all_nodes if node.get("isClickable", False)]

        return filter_fn

    @staticmethod
    def non_clickable() -> ElementFilter:
        """Match non-clickable elements."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [node for node in all_nodes if not node.get("isClickable", False)]

        return filter_fn

    @staticmethod
    def enabled(expected: bool = True) -> ElementFilter:
        """Match elements by enabled state."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [
                node for node in all_nodes if node.get("isEnabled", False) == expected
            ]

        return filter_fn

    @staticmethod
    def selected(expected: bool = True) -> ElementFilter:
        """Match elements by selected state."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [
                node for node in all_nodes if node.get("isSelected", False) == expected
            ]

        return filter_fn

    @staticmethod
    def checked(expected: bool = True) -> ElementFilter:
        """Match elements by checked state."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [
                node for node in all_nodes if node.get("isChecked", False) == expected
            ]

        return filter_fn

    @staticmethod
    def focused(expected: bool = True) -> ElementFilter:
        """Match elements by focused state."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [
                node for node in all_nodes if node.get("isFocused", False) == expected
            ]

        return filter_fn

    # ========== SIZE MATCHING ==========

    @staticmethod
    def size_matches(
        width: int = None, height: int = None, tolerance: int = 0
    ) -> ElementFilter:
        """Match elements by size (width and/or height with tolerance)."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            results = []
            for node in all_nodes:
                bounds = node.get("boundsInScreen", {})

                actual_width = bounds.get("right", 0) - bounds.get("left", 0)
                actual_height = bounds.get("bottom", 0) - bounds.get("top", 0)

                if width is not None:
                    if abs(actual_width - width) > tolerance:
                        continue

                if height is not None:
                    if abs(actual_height - height) > tolerance:
                        continue

                results.append(node)

            return results

        return filter_fn

    # ========== HIERARCHY FILTERS ==========

    @staticmethod
    def contains_child(child_filter: ElementFilter) -> ElementFilter:
        """Match elements that contain at least one direct child matching the filter."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            results = []

            for node in all_nodes:
                children = node.get("children", [])

                if not children:
                    continue

                matching_children = child_filter(children)

                if matching_children:
                    results.append(node)

            return results

        return filter_fn

    @staticmethod
    def contains_descendants(filters: List[ElementFilter]) -> ElementFilter:
        """Match elements that contain ALL specified descendants at any depth."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            results = []

            for node in all_nodes:
                descendants = flatten_tree(node)

                all_match = True

                for descendant_filter in filters:
                    matches = descendant_filter(descendants)

                    if not matches:
                        all_match = False
                        break

                if all_match:
                    results.append(node)

            return results

        return filter_fn

    @staticmethod
    def child_of(parent_filter: ElementFilter) -> ElementFilter:
        """Match elements that are direct children of elements matching the parent filter."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            parents = parent_filter(nodes)

            if not parents:
                return []

            results = []
            for parent in parents:
                results.extend(parent.get("children", []))

            return results

        return filter_fn

    # ========== UTILITY FILTERS ==========

    @staticmethod
    def has_text() -> ElementFilter:
        """Match elements that have non-empty text content."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [
                node
                for node in all_nodes
                if (
                    node.get("text")
                    or node.get("contentDescription")
                    or node.get("hint")
                )
            ]

        return filter_fn

    @staticmethod
    def clickable_first() -> ElementFilter:
        """Sort elements to put clickable ones first."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return sorted(all_nodes, key=lambda n: not n.get("isClickable", False))

        return filter_fn

    # ========== INDEX SELECTION ==========

    @staticmethod
    def index(idx: int) -> ElementFilter:
        """Select element at index position (supports negative indices)."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            sorted_nodes = sort_by_position(all_nodes)

            try:
                return [sorted_nodes[idx]]
            except IndexError:
                return []

        return filter_fn

    # ========== COMPOSITION ==========

    @staticmethod
    def compose(filters: List[ElementFilter]) -> ElementFilter:
        """Apply filters sequentially (pipeline)."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            result = nodes

            for f in filters:
                result = f(result)

                if not result:
                    break

            return result

        return filter_fn

    @staticmethod
    def intersect(filters: List[ElementFilter]) -> ElementFilter:
        """Return elements matching ALL filters (AND logic)."""

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            if not filters:
                return nodes

            result_sets = [set(id(n) for n in filters[0](nodes))]

            for f in filters[1:]:
                result_ids = set(id(n) for n in f(nodes))
                result_sets.append(result_ids)

            common_ids = result_sets[0].intersection(*result_sets[1:])

            all_nodes = []
            for node in nodes:
                all_nodes.extend(flatten_tree(node))

            return [n for n in all_nodes if id(n) in common_ids]

        return filter_fn

    # ========== DEEPEST MATCHING ==========

    @staticmethod
    def deepest_matching(base_filter: ElementFilter) -> ElementFilter:
        """Find the deepest (most specific) matching elements in the tree."""

        def find_deepest_in_node(node: Dict) -> List[Dict]:
            child_matches = []
            for child in node.get("children", []):
                child_matches.extend(find_deepest_in_node(child))

            if child_matches:
                return child_matches

            current_matches = base_filter([node])
            if current_matches:
                return [node]

            return []

        def filter_fn(nodes: List[Dict]) -> List[Dict]:
            results = []

            for node in nodes:
                results.extend(find_deepest_in_node(node))

            return results

        return filter_fn
