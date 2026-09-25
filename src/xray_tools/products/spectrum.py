"""File-backed spectral products from CIAO's specextract script."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import CIAOWorkflowError
from ..utils import require_file, require_pycrates, require_sherpa

if TYPE_CHECKING:
    from sherpa.astro.data import DataPHA


@dataclass(frozen=True)
class CIAOSpectrum:
    """File-backed specextract products for one source extraction.

    ``source`` is the ungrouped PI spectrum; the remaining fields are present
    only when specextract created them (background input, grouping, background
    responses, PSF-corrected ARF). Construction is free; header metadata is
    read through pycrates on first access.
    """

    source: Path
    arf: Path | None = None
    rmf: Path | None = None
    background: Path | None = None
    grouped_source: Path | None = None
    grouped_background: Path | None = None
    bkg_arf: Path | None = None
    bkg_rmf: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        for field in (
            "arf",
            "rmf",
            "background",
            "grouped_source",
            "grouped_background",
            "bkg_arf",
            "bkg_rmf",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, Path(value))

    def _crate(self) -> Any:
        require_file(self.source, "PHA spectrum")
        return require_pycrates().read_file(str(self.source))

    @cached_property
    def exposure(self) -> float | None:
        """The EXPOSURE keyword of the source spectrum, in seconds."""
        value = self._crate().get_key_value("EXPOSURE")
        return float(value) if value is not None else None

    @cached_property
    def backscal(self) -> float | None:
        """The BACKSCAL keyword of the source spectrum (region area)."""
        value = self._crate().get_key_value("BACKSCAL")
        return float(value) if value is not None else None

    @cached_property
    def counts(self) -> int | None:
        """Total source counts from the TOTCTS keyword, when recorded."""
        value = self._crate().get_key_value("TOTCTS")
        return int(value) if value is not None else None

    def to_sherpa(
        self,
        *,
        prefer_grouped: bool = True,
        use_errors: bool = False,
    ) -> DataPHA:
        """Read the spectrum into a ``sherpa.astro.data.DataPHA``.

        Reads the grouped spectrum when grouping was applied (and
        ``prefer_grouped`` is left at its default), otherwise the ungrouped
        one. Sherpa auto-attaches the ARF, RMF, and background listed in the
        PHA header keywords, which specextract maintains.
        """
        path = self.source
        if prefer_grouped and self.grouped_source is not None:
            path = self.grouped_source
        require_file(path, "PHA spectrum")
        return require_sherpa().read_pha(str(path), use_errors=use_errors)


def _discover_spectrum(outroot: str | Path, *, since: float) -> CIAOSpectrum:
    """Collect the specextract products of one extraction created after ``since``.

    Follows the specextract naming for a single input, where ``outroot`` is a
    plain file prefix: ``{outroot}.pi`` / ``{outroot}.arf`` / ``{outroot}.rmf``
    (plus ``{outroot}.corr.arf`` when the PSF correction ran and
    ``{outroot}_grp.pi`` when grouped) and ``{outroot}_bkg.*`` for the
    background. Only files newer than the given timestamp are considered, so
    products from earlier runs with the same outroot are left out.
    """
    raw = str(outroot)
    base = Path(raw).parent
    src = Path(raw).name
    bkg = f"{src}_bkg"

    def fresh(path: Path) -> bool:
        return path.is_file() and path.stat().st_mtime >= since - 1.0

    def first_fresh(*names: str) -> Path | None:
        for name in names:
            candidate = base / name
            if fresh(candidate):
                return candidate
        return None

    source = first_fresh(f"{src}.pi")
    if source is None:
        raise CIAOWorkflowError(
            f"specextract created no source spectrum for outroot: {outroot}"
        )
    return CIAOSpectrum(
        source=source,
        arf=first_fresh(f"{src}.corr.arf", f"{src}.arf"),
        rmf=first_fresh(f"{src}.rmf"),
        background=first_fresh(f"{bkg}.pi"),
        grouped_source=first_fresh(f"{src}_grp.pi"),
        grouped_background=first_fresh(f"{bkg}_grp.pi"),
        bkg_arf=first_fresh(f"{bkg}.arf"),
        bkg_rmf=first_fresh(f"{bkg}.rmf"),
    )
