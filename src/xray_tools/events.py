"""Lazy CIAO Data Model views over Chandra event files."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import CIAOWorkflowError
from .products.image import _discover_flux_images
from .utils import (
    prepare_output_path,
    require_ciao_runtool,
    require_file,
    require_pycrates,
)

if TYPE_CHECKING:
    import pandas as pd

    from .products import CIAOGTI, CIAOBlankSky, CIAOFluxImage, CIAOLightCurve


logger = logging.getLogger("xray_tools")


@dataclass(frozen=True)
class CIAOEvents:
    """An immutable, lazy selection over one physical event file."""

    filename: Path
    filters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "filename", Path(self.filename))
        object.__setattr__(self, "filters", tuple(self.filters))

    def filter(self, expression: str) -> CIAOEvents:
        """Return a new event view with a raw Data Model filter appended."""
        return CIAOEvents(self.filename, self.filters + (expression,))

    def energy(self, emin_ev: float, emax_ev: float) -> CIAOEvents:
        """Select an energy interval expressed in electron volts."""
        if not emin_ev < emax_ev:
            raise ValueError("emin_ev must be smaller than emax_ev")

        return self.filter(f"energy={emin_ev}:{emax_ev}")

    def ccd_id(self, *ccd_ids: int) -> CIAOEvents:
        """Select events from specific CCD IDs."""
        if not ccd_ids:
            raise ValueError("ccd_ids must contain at least one CCD ID")
        return self.filter(f"ccd_id={','.join(map(str, ccd_ids))}")

    def region(self, region_file: str | Path) -> CIAOEvents:
        """Select events inside a CIAO-compatible region file."""
        path = Path(region_file)
        require_file(path, "region file")
        return self.filter(f"sky=region({path})")

    def exclude_region(self, region_file: str | Path) -> CIAOEvents:
        """Exclude events inside a CIAO-compatible region file."""
        path = Path(region_file)
        require_file(path, "region file")
        return self.filter(f"exclude sky=region({path})")

    def apply_gti(self, gti: CIAOGTI | str | Path) -> CIAOEvents:
        """Apply a GTI file as a lazy CIAO ``@`` filter."""
        from .products import CIAOGTI

        path = gti.path if isinstance(gti, CIAOGTI) else Path(gti)
        require_file(path, "GTI file")
        return self.filter(f"@{path}")

    @property
    def dm(self) -> str:
        """The complete CIAO Data Model expression for this event view."""
        return str(self.filename) + _format_filters(self.filters)

    def write(
        self,
        outfile: str | Path,
        *,
        clobber: bool = False,
    ) -> CIAOEvents:
        """Materialize this event view with ``dmcopy``."""
        require_file(self.filename, "event file")
        output = prepare_output_path(outfile, clobber=clobber)
        runtool = require_ciao_runtool()

        logger.info("dmcopy: %s", self.dm)
        logger.info("output: %s", output)
        runtool.dmcopy(
            infile=self.dm,
            outfile=str(output),
            clobber=clobber,
        )

        return CIAOEvents(output)

    def to_crate(self) -> Any:
        """Read this event view through pycrates without materializing it."""
        require_file(self.filename, "event file")
        return require_pycrates().read_file(self.dm)

    def to_pandas(
        self,
        columns: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Copy event columns from pycrates into a pandas DataFrame."""
        import pandas as pd

        require_file(self.filename, "event file")
        if columns is not None and not columns:
            raise ValueError("columns must contain at least one column name")

        expression = self.dm
        if columns is not None:
            expression += f"[cols {','.join(columns)}]"

        crate = require_pycrates().read_file(expression)
        requested = columns
        if requested is None:
            requested = tuple(
                component
                for name in crate.get_colnames()
                for component in _column_components(name)
            )

        data = {}
        for name in requested:
            try:
                values = crate.get_column(name).values.copy()
            except ValueError as exc:
                raise KeyError(f"column not found in event data: {name}") from exc

            # pandas requires one-dimensional values per DataFrame column.
            data[name.lower()] = values if values.ndim == 1 else list(values)

        return pd.DataFrame(data)

    def lightcurve(
        self,
        outfile: str | Path,
        *,
        binsize: float = 259.28,
        clobber: bool = False,
    ) -> CIAOLightCurve:
        """Extract a file-backed ``ltc1`` light curve with ``dmextract``."""
        from .products import CIAOLightCurve

        if not binsize > 0:
            raise ValueError("binsize must be positive")

        require_file(self.filename, "event file")
        output = prepare_output_path(outfile, clobber=clobber, create_parents=True)
        infile = f"{self.dm}[bin time=::{binsize}]"
        runtool = require_ciao_runtool()

        logger.info("dmextract: %s", infile)
        logger.info("output: %s", output)
        runtool.dmextract(
            infile=infile,
            outfile=str(output),
            opt="ltc1",
            clobber=clobber,
        )

        return CIAOLightCurve(output)

    def fluximage(
        self,
        outroot: str | Path,
        *,
        energy: tuple[float, float] | list[tuple[float, float]] | None = None,
        bands: str | None = None,
        binsize: float | None = None,
        xygrid: str | None = None,
        units: str = "default",
        expmapthresh: str = "1.5%",
        psfecf: float | None = None,
        asolfile: str | None = None,
        badpixfile: str | None = None,
        maskfile: str | None = None,
        dtffile: str | None = None,
        background: str | None = None,
        parallel: bool = True,
        nproc: int | None = None,
        cleanup: bool = True,
        verbose: int = 1,
        clobber: bool = False,
    ) -> dict[str, CIAOFluxImage]:
        """Create exposure-corrected images with CIAO's ``fluximage`` script.

        ``energy`` selects one or more custom keV intervals; each interval is
        serialized to CIAO's ``lo:hi:eff`` syntax with the effective energy
        set to the geometric mean of the bounds. ``bands`` passes CIAO's band
        syntax through verbatim (band names such as ``"csc"``, explicit
        ``"lo:hi:eff"`` ranges, or spectral-weights files). The two options
        are mutually exclusive.

        The lazy event view is used directly as the input, so filters apply
        to the images as well. ``outroot`` follows the fluximage convention:
        a trailing ``/`` writes into that directory, otherwise it is a file
        prefix; missing directories are created. Returns the products created
        by this run, mapped by band label.
        """
        require_file(self.filename, "event file")
        if energy is not None and bands is not None:
            raise ValueError("pass either energy or bands, not both")
        bands_value = _serialize_energy_bands(energy) if energy is not None else bands

        raw = str(outroot)
        if raw.endswith("/"):
            Path(raw).mkdir(parents=True, exist_ok=True)
        else:
            Path(raw).parent.mkdir(parents=True, exist_ok=True)

        options: dict[str, Any] = {}
        if bands_value is not None:
            options["bands"] = bands_value
        if binsize is not None:
            options["binsize"] = binsize
        if xygrid is not None:
            options["xygrid"] = xygrid
        if psfecf is not None:
            options["psfecf"] = psfecf
        if asolfile is not None:
            options["asolfile"] = asolfile
        if badpixfile is not None:
            options["badpixfile"] = badpixfile
        if maskfile is not None:
            options["maskfile"] = maskfile
        if dtffile is not None:
            options["dtffile"] = dtffile
        if background is not None:
            options["background"] = background
        if nproc is not None:
            options["nproc"] = nproc

        runtool = require_ciao_runtool()
        logger.info("fluximage: %s", self.dm)
        logger.info("outroot: %s", raw)
        started = time.time()
        runtool.fluximage(
            infile=self.dm,
            outroot=raw,
            units=units,
            expmapthresh=expmapthresh,
            parallel=parallel,
            cleanup=cleanup,
            verbose=verbose,
            clobber=clobber,
            **options,
        )

        images = _discover_flux_images(raw, since=started)
        if not images:
            raise CIAOWorkflowError(f"fluximage created no flux images for outroot: {raw}")
        return images

    def blanksky(
        self,
        outfile: str | Path,
        *,
        asolfile: str | Path | Sequence[str | Path] | None = None,
        stowedbg: bool = False,
        weight_method: str = "particle",
        bkgparams: str = "default",
        random: int = 0,
        tmpdir: str | Path | None = None,
        verbose: int = 0,
        clobber: bool = False,
    ) -> CIAOBlankSky:
        """Tailor a CalDB blank-sky background event file to this observation.

        ``weight_method`` selects the background normalization: ``"particle"``
        (default) scales by high-energy particle counts, ``"time"`` by exposure
        time, and ``"particle-rate"`` by particle count rates. ``bkgparams`` is
        the Data Model filter used for the particle-based methods — ``"default"``
        lets CIAO pick (``[energy=9000:12000]`` for ACIS, ``[pi=300:500]`` for
        HRC) — or an explicit ``"[energy=lo:hi]"`` / ``"[pi=lo:hi]"`` filter.
        ``asolfile`` may be a path or a sequence of paths; when omitted, CIAO
        looks the aspect solution up from the event file header. Requires the
        CalDB blank-sky background files to be installed.
        """
        from .products.blanksky import CIAOBlankSky

        methods = ("particle", "time", "particle-rate")
        if weight_method not in methods:
            raise ValueError(f"weight_method must be one of {sorted(methods)}")
        if bkgparams != "default" and re.fullmatch(
            r"\[(energy|pi)=\d+:\d+\]", bkgparams
        ) is None:
            raise ValueError(
                'bkgparams must be "default" or a filter like "[energy=9000:12000]"'
            )

        require_file(self.filename, "event file")
        output = prepare_output_path(outfile, clobber=clobber, create_parents=True)

        options: dict[str, Any] = {}
        if asolfile is not None:
            if isinstance(asolfile, (str, Path)):
                asol_files = (asolfile,)
            else:
                asol_files = tuple(asolfile)
            for path in asol_files:
                require_file(Path(path), "aspect solution file")
            options["asolfile"] = ",".join(str(path) for path in asol_files)
        if tmpdir is not None:
            options["tmpdir"] = str(tmpdir)

        runtool = require_ciao_runtool()
        logger.info("blanksky: %s", self.dm)
        logger.info("output: %s", output)
        runtool.blanksky(
            evtfile=self.dm,
            outfile=str(output),
            stowedbg=stowedbg,
            weight_method=weight_method,
            bkgparams=bkgparams,
            random=random,
            verbose=verbose,
            clobber=clobber,
            **options,
        )
        return CIAOBlankSky(output)


def merge_events(
    events: Sequence[CIAOEvents | str | Path],
    outfile: str | Path,
    *,
    clobber: bool = False,
) -> CIAOEvents:
    """Merge event files or lazy event views into one file with ``dmmerge``.

    Inputs are merged in order. Lazy :class:`CIAOEvents` views contribute
    their Data Model expression, so their filters apply to the merged
    output. Returns the merged file as a new :class:`CIAOEvents`.
    """
    if not events:
        raise ValueError("events must contain at least one input")

    infiles = []
    for item in events:
        if isinstance(item, CIAOEvents):
            require_file(item.filename, "event file")
            infiles.append(item.dm)
        else:
            path = Path(item)
            require_file(path, "event file")
            infiles.append(str(path))

    output = prepare_output_path(outfile, clobber=clobber, create_parents=True)
    runtool = require_ciao_runtool()
    infile = ",".join(infiles)

    logger.info("dmmerge: %s", infile)
    logger.info("output: %s", output)
    runtool.dmmerge(
        infile=infile,
        outfile=str(output),
        clobber=clobber,
    )
    return CIAOEvents(output)


def _serialize_energy_bands(
    energy: tuple[float, float] | list[tuple[float, float]],
) -> str:
    """Serialize custom keV interval(s) to CIAO's ``lo:hi:eff`` band syntax.

    The effective energy is the geometric mean of the interval bounds.
    """
    intervals = [energy] if isinstance(energy, tuple) else list(energy)
    if not intervals:
        raise ValueError("energy must contain at least one (lo, hi) interval")

    parts = []
    for interval in intervals:
        try:
            lo, hi = interval
        except (TypeError, ValueError) as exc:
            raise ValueError(f"energy intervals must be (lo, hi) pairs: {interval!r}") from exc
        if not 0 < lo < hi:
            raise ValueError(f"energy intervals must satisfy 0 < lo < hi: {interval!r}")
        eff = (lo * hi) ** 0.5
        parts.append(f"{lo:g}:{hi:g}:{eff:.6g}")
    return ",".join(parts)


def _column_components(name: str) -> tuple[str, ...]:
    """Expand a pycrates vector label such as ``sky(x, y)``."""
    start = name.find("(")
    if start == -1 or not name.endswith(")"):
        return (name,)

    return tuple(component.strip() for component in name[start + 1 : -1].split(","))


def _invert_filter(expression: str) -> str:
    """Return the complement of a supported CIAO include filter.

    Returns an empty string when the complement selects nothing, e.g. all
    CCD IDs are already selected. Raises :class:`CIAOWorkflowError` for
    filters whose complement cannot be expressed as a single CIAO filter.
    """
    if expression.startswith("energy="):
        bounds = expression.removeprefix("energy=").split(":")
        if len(bounds) != 2 or not all(bounds):
            raise CIAOWorkflowError(f"cannot invert energy filter: {expression}")

        lower, upper = bounds

        # energy=500:7000  ->  energy=:500,7000:
        return f"energy=:{lower},{upper}:"

    if expression.startswith("ccd_id="):
        try:
            selected = {
                int(value) for value in expression.removeprefix("ccd_id=").split(",")
            }
        except ValueError as exc:
            raise CIAOWorkflowError(f"cannot invert CCD filter: {expression}") from exc

        unselected = sorted(set(range(10)) - selected)
        if not unselected:
            return ""
        return f"ccd_id={','.join(map(str, unselected))}"

    raise CIAOWorkflowError(
        "CIAO cannot compose an excluded region with filter "
        f"{expression!r}. Materialize that selection with write() "
        "before applying exclude_region()."
    )


def _format_filters(filters: tuple[str, ...]) -> str:
    """Serialize a set of lazy CIAO filters.

    CIAO cannot mix normal filters with ``exclude`` filters directly.
    When exclusions are present, rewrite

        include A AND include B AND NOT C

    as

        NOT (NOT A OR NOT B OR C)

    and express the result as one CIAO ``exclude`` filter.
    """
    excluded = [
        expression.removeprefix("exclude ")
        for expression in filters
        if expression.startswith("exclude ")
    ]
    included = [
        expression for expression in filters if not expression.startswith("exclude ")
    ]

    # Normal case: no exclusions, so filters can simply be chained.
    if not excluded:
        return "".join(f"[{expression}]" for expression in included)

    # A lone exclusion needs no complement rewrite.
    if not included and len(excluded) == 1:
        return f"[exclude {excluded[0]}]"

    # Convert included selections into events that must be excluded;
    # an inversion may select nothing (all CCD IDs included) and is dropped.
    inverted = [_invert_filter(expression) for expression in included]
    terms = excluded + [term for term in inverted if term]

    compound = "||".join(f"({term})" for term in terms)
    return f"[exclude {compound}]"
