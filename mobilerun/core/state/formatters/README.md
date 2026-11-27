# core/state/formatters/

Formats filtered UI trees with indexed elements.

Keep tree-to-text formatters here; filtering belongs in `core/state/filters/`.

## Files

- `__init__.py` — Exports `TreeFormatter`, `IndexedFormatter`.
- `base.py` — `TreeFormatter` — base interface for tree formatters.
- `indexed_formatter.py` — `IndexedFormatter` — indexed formatter - standard mobilerun format.

**Depends on / used by:** Uses coordinate and geometry utilities; used by Android and iOS state providers and device defaults.

See [state](../README.md) and [filters](../filters/README.md).
