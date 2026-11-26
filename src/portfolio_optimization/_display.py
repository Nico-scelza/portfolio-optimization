"""Optional IPython display helpers with headless fallback and concurrent isolation."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

try:
    from IPython.display import (
        HTML as _IPYTHON_HTML,
        Markdown as _IPYTHON_MARKDOWN,
        clear_output as _IPYTHON_CLEAR_OUTPUT,
        display as _IPYTHON_DISPLAY,
    )

    HAS_IPYTHON = True
except Exception:  # pragma: no cover - headless environments without IPython
    _IPYTHON_HTML = None
    _IPYTHON_MARKDOWN = None
    _IPYTHON_CLEAR_OUTPUT = None
    _IPYTHON_DISPLAY = None
    HAS_IPYTHON = False

_DISPLAY_ENABLED_VAR: ContextVar[bool] = ContextVar("portfolio_display_enabled", default=True)


class _HeadlessContent:
    """Minimal placeholder wrapper when IPython is unavailable."""

    def __init__(self, data: Any) -> None:
        self.data = data

    def __str__(self) -> str:
        return str(self.data)


def HTML(data: Any = None, **kwargs: Any) -> Any:
    if _IPYTHON_HTML is not None:
        return _IPYTHON_HTML(data, **kwargs)
    return _HeadlessContent(data)


def Markdown(data: Any = None, **kwargs: Any) -> Any:
    if _IPYTHON_MARKDOWN is not None:
        return _IPYTHON_MARKDOWN(data, **kwargs)
    return _HeadlessContent(data)


def is_display_enabled() -> bool:
    """Return whether IPython display is enabled in the current thread/coroutine context."""
    return bool(_DISPLAY_ENABLED_VAR.get())


def display(*objs: Any, **kwargs: Any) -> None:
    if not _DISPLAY_ENABLED_VAR.get():
        return
    if _IPYTHON_DISPLAY is not None:
        _IPYTHON_DISPLAY(*objs, **kwargs)


def clear_output(wait: bool = False) -> None:
    if not _DISPLAY_ENABLED_VAR.get():
        return
    if _IPYTHON_CLEAR_OUTPUT is not None:
        _IPYTHON_CLEAR_OUTPUT(wait=wait)


def set_display_enabled(enabled: bool) -> None:
    """Set display flag strictly for the current execution context (ContextVar)."""
    _DISPLAY_ENABLED_VAR.set(bool(enabled))


@contextmanager
def suppress_display() -> Iterator[None]:
    """Temporarily silence IPython display calls in the current thread/task only.

    Uses ``contextvars.ContextVar`` with token reset so concurrent threads and
    asyncio coroutines are completely isolated and nested contexts restore cleanly.
    """
    token = _DISPLAY_ENABLED_VAR.set(False)
    try:
        yield
    finally:
        _DISPLAY_ENABLED_VAR.reset(token)