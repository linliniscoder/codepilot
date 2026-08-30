from __future__ import annotations

import json
import re
from typing import Any


INVALID_ESCAPE = re.compile(r"\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})")


def loads_with_repaired_escapes(text: str) -> Any:
    """Parse model JSON and repair raw regex escapes such as ``\\s``."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        repaired = INVALID_ESCAPE.sub(lambda _: "\\\\", text)
        if repaired == text:
            raise
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise original_error
