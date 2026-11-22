"""Pure verification-code extraction."""

import re

_CODE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")


def extract_code(text: str | None) -> str | None:
    match = _CODE.search(text or "")
    return match.group() if match else None
