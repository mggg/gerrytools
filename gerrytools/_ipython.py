"""Internal IPython/Jupyter environment detection.

Single home for the "are we inside a live Jupyter kernel?" check used by the
plotting and latex modules to decide whether to suppress implicit figure
display or render inline output.
"""

from __future__ import annotations


def in_jupyter_kernel() -> bool:
    """Return True iff running inside a live Jupyter kernel.

    A plain IPython terminal session has an interactive shell but no
    ``kernel`` attribute, so this returns False there — as it does in any
    ordinary Python process, including environments where IPython is not
    installed.
    """
    try:  # pragma: no cover - only reachable inside a live IPython session
        from IPython.core.getipython import get_ipython

        interactive_shell = get_ipython()
    except Exception:  # pragma: no cover - IPython missing or shell lookup failed
        return False
    return interactive_shell is not None and getattr(interactive_shell, "kernel", None) is not None
