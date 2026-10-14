"""File-backed blank-sky background products from CIAO's blanksky scripts."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from ..exceptions import CIAOWorkflowError
from ..utils import (
    prepare_output_path,
    require_ciao_runtool,
    require_file,
    require_pycrates,
)
from .image import CIAOImage

_SCAL_KEY = re.compile(r"^BKGSCAL(\d+)$")


@dataclass(frozen=True)
class CIAOBlankSky:
    """A reference to a tailored blank-sky background event file.

    The file is produced by CIAO's ``blanksky`` script for one observation and
    carries the ``BKGMETH`` / ``BKGPARAM`` keywords plus per-chip ``BKGSCALn``
    scaling factors (ACIS). Construction is free; metadata is read through
    pycrates on first access.
    """

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def _crate(self) -> Any:
        require_file(self.path, "blanksky background file")
        return require_pycrates().read_file(str(self.path))

    @cached_property
    def method(self) -> str:
        """The background scaling method: particle, time, or particle-rate."""
        value = self._crate().get_key_value("BKGMETH")
        if value is None:
            raise ValueError(f"not a tailored blanksky file (no BKGMETH): {self.path}")
        return str(value)

    @cached_property
    def bkgparams(self) -> str | None:
        """The Data Model filter used for the particle scaling, when recorded."""
        value = self._crate().get_key_value("BKGPARAM")
        return str(value) if value is not None else None

    @cached_property
    def scales(self) -> dict[int, float]:
        """Per-chip scaling factors keyed by CCD ID (ACIS ``BKGSCALn``).

        Single-chip instruments (HRC) record a plain ``BKGSCALE`` instead, so
        their files yield an empty dict.
        """
        crate = self._crate()
        return {
            int(match.group(1)): float(crate.get_key_value(name))
            for name in crate.get_keynames()
            if (match := _SCAL_KEY.match(name))
        }

    def images(
        self,
        outroot: str | Path,
        reference: str | Path | CIAOImage,
        *,
        tmpdir: str | Path | None = None,
        verbose: int = 0,
        clobber: bool = False,
    ) -> CIAOBlankSkyImage:
        """Create scaled background and background-subtracted images.

        ``reference`` is the counts image the background is scaled to — a
        :class:`CIAOImage` (for example the ``counts`` component of a fluximage
        band) or a path to one. It defines the grid, binning, and energy band
        of the outputs. ``outroot`` is a file prefix; the products are named
        ``{outroot}_{method}_bgnd.img`` and ``{outroot}_{method}_bkgsub.img``.
        """
        require_file(self.path, "blanksky background file")
        if isinstance(reference, CIAOImage):
            reference_path = Path(reference.path)
        else:
            reference_path = Path(reference)
        require_file(reference_path, "reference image")
        prepare_output_path(
            f"{outroot}_{self.method}_bgnd.img",
            clobber=clobber,
            create_parents=True,
        )

        options: dict[str, Any] = {}
        if tmpdir is not None:
            options["tmpdir"] = str(tmpdir)

        runtool = require_ciao_runtool()
        started = time.time()
        runtool.blanksky_image(
            bkgfile=str(self.path),
            outroot=str(outroot),
            imgfile=str(reference_path),
            verbose=verbose,
            clobber=clobber,
            **options,
        )

        products = _discover_blanksky_images(outroot, since=started)
        return CIAOBlankSkyImage(
            background=products["bgnd"],
            subtracted=products["bkgsub"],
        )


@dataclass(frozen=True)
class CIAOBlankSkyImage:
    """Scaled background and background-subtracted images for one band."""

    background: CIAOImage
    subtracted: CIAOImage

    def __post_init__(self) -> None:
        for field in ("background", "subtracted"):
            value = getattr(self, field)
            if not isinstance(value, CIAOImage):
                object.__setattr__(self, field, CIAOImage(value))


def _discover_blanksky_images(outroot: str | Path, *, since: float) -> dict[str, Path]:
    """Map bgnd/bkgsub suffixes to the images created after ``since``.

    Only files newer than the given timestamp are considered, so products
    from earlier runs with the same outroot are left out.
    """
    base = Path(str(outroot)).parent
    prefix = Path(str(outroot)).name

    products: dict[str, Path] = {}
    for suffix in ("bgnd", "bkgsub"):
        candidates = sorted(
            path
            for path in base.glob(f"{prefix}_*_{suffix}.img")
            if path.is_file() and path.stat().st_mtime >= since - 1.0
        )
        if not candidates:
            raise CIAOWorkflowError(
                f"blanksky_image created no {suffix} image for outroot: {outroot}"
            )
        products[suffix] = candidates[0]
    return products
