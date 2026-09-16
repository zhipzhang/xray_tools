"""Lazy CIAO Data Model views over Chandra event files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .exceptions import CIAOWorkflowError
from .utils import (
    prepare_output_path,
    require_ciao_runtool,
    require_file,
    require_pycrates,
)

if TYPE_CHECKING:
    import pandas as pd

    from .products import CIAOGTI, CIAOLightCurve


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
        output = prepare_output_path(outfile, clobber=clobber)
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


def _column_components(name: str) -> tuple[str, ...]:
    """Expand a pycrates vector label such as ``sky(x, y)``."""
    start = name.find("(")
    if start == -1 or not name.endswith(")"):
        return (name,)

    return tuple(component.strip() for component in name[start + 1 : -1].split(","))


def _format_filters(filters: tuple[str, ...]) -> str:
    """Serialize filters, accounting for CIAO's exclude-filter restriction."""
    excluded = tuple(
        expression.removeprefix("exclude ")
        for expression in filters
        if expression.startswith("exclude ")
    )
    if not excluded:
        return "".join(f"[{expression}]" for expression in filters)

    included = tuple(
        expression for expression in filters if not expression.startswith("exclude ")
    )
    if not included and len(excluded) == 1:
        return f"[exclude {excluded[0]}]"

    terms = list(excluded)
    for expression in included:
        if expression.startswith("energy="):
            bounds = expression.removeprefix("energy=").split(":")
            if len(bounds) != 2 or not all(bounds):
                raise CIAOWorkflowError(
                    f"cannot compose energy filter with region exclusion: {expression}"
                )
            lower, upper = bounds
            terms.append(f"energy=:{lower},{upper}:")
            continue

        if expression.startswith("ccd_id="):
            try:
                selected = {
                    int(value)
                    for value in expression.removeprefix("ccd_id=").split(",")
                }
            except ValueError as exc:
                raise CIAOWorkflowError(
                    f"cannot compose CCD filter with region exclusion: {expression}"
                ) from exc

            unselected = sorted(set(range(10)) - selected)
            if unselected:
                terms.append(f"ccd_id={','.join(map(str, unselected))}")
            continue

        raise CIAOWorkflowError(
            "CIAO cannot compose an excluded region with filter "
            f"{expression!r}. Materialize that selection with write() "
            "before applying exclude_region()."
        )

    compound = "||".join(f"({term})" for term in terms)
    return f"[exclude {compound}]"
