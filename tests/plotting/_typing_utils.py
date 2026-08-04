"""Static-typing helpers for the plotting test suite."""

from typing import Any


def as_any(value: object) -> Any:
    """Identity function that erases the static type of ``value``.

    Tests deliberately pass values outside a parameter's declared type to exercise runtime
    validation and coercion. Routing them through this launder keeps the call sites clean under
    pyright and ty without per-line suppressions, while leaving runtime behavior untouched.
    """
    return value
