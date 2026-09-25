"""Internal dependency and path helpers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .exceptions import CIAOWorkflowError


def require_ciao_runtool() -> Any:
    """Return CIAO's runtool module or raise an actionable error."""
    try:
        from ciao_contrib import runtool
    except ImportError as exc:
        raise CIAOWorkflowError(
            "CIAO ciao_contrib.runtool is not available. "
            "Run this operation inside a configured CIAO environment."
        ) from exc

    return runtool


def require_pycrates() -> Any:
    """Return pycrates or raise an actionable error."""
    try:
        import pycrates
    except ImportError as exc:
        raise CIAOWorkflowError(
            "CIAO pycrates is not available. "
            "Run this operation inside a configured CIAO environment."
        ) from exc

    return pycrates


def require_lc_clean() -> Callable[..., Any]:
    """Return CIAO's lc_clean function or raise an actionable error."""
    try:
        from lightcurves import lc_clean
    except ImportError as exc:
        raise CIAOWorkflowError(
            "CIAO lightcurves.lc_clean is not available. "
            "Run this operation inside a configured CIAO environment."
        ) from exc

    return lc_clean


def require_sherpa() -> Any:
    """Return sherpa's astro I/O module or raise an actionable error."""
    try:
        from sherpa.astro import io
    except ImportError as exc:
        raise CIAOWorkflowError(
            "sherpa is not available. "
            "Run this operation inside a configured CIAO environment."
        ) from exc

    return io


def require_file(path: Path, description: str) -> None:
    """Require a regular input file before invoking CIAO."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def prepare_output_path(
    outfile: str | Path,
    *,
    clobber: bool,
    create_parents: bool = False,
) -> Path:
    """Validate an output location, optionally creating missing directories."""
    path = Path(outfile)
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    if path.exists() and not clobber:
        raise FileExistsError(f"output file already exists: {path}")
    return path
