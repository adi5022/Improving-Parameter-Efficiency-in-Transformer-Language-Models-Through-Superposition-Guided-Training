import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def impl(fn, *args, **kwargs):
    """Call user-written code; skip (not fail) while it is still a NotImplementedError stub."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")
