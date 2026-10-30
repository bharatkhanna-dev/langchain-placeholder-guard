"""Model-agnostic LangChain middleware that guards against unfilled placeholders."""

from .detection import (
    DEFAULT_PLACEHOLDER_PATTERNS,
    OPTIONAL_PATTERNS,
    Finding,
    find_placeholders,
)
from .middleware import (
    DEFAULT_PLACEHOLDER_NUDGE,
    PlaceholderGuardMiddleware,
    PlaceholderLeakError,
)

__all__ = [
    "DEFAULT_PLACEHOLDER_NUDGE",
    "DEFAULT_PLACEHOLDER_PATTERNS",
    "OPTIONAL_PATTERNS",
    "Finding",
    "PlaceholderGuardMiddleware",
    "PlaceholderLeakError",
    "find_placeholders",
]

__version__ = "0.1.0"
