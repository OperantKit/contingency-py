"""contingency — Python reinforcement schedule engine.

Bridges contingency-dsl AST nodes into executable Schedule runtime objects.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

try:
    from contingency._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"
