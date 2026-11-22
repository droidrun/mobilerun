# core/state/filters/

Filters accessibility trees before state formatting.

Keep tree-filter strategies here; rendering belongs in `core/state/formatters/`.

## Files

- `__init__.py` — Exports `TreeFilter`, `ConciseFilter`, `DetailedFilter`, `get_filter`.
- `base.py` — `TreeFilter` — base interface for accessibility tree filters.
- `concise_filter.py` — `ConciseFilter` — concise filtering - all logic self-contained.
- `detailed_filter.py` — `DetailedFilter` — detailed filtering - all logic self-contained.
- `factory.py` — `get_filter` — ui tree filter factory.

**Depends on / used by:** Uses the tree-filter interface; used by Android and iOS state providers and device defaults.

See [state](../README.md) and [formatters](../formatters/README.md).
