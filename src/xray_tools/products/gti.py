"""File-backed CIAO good-time interval products."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import require_file, require_pycrates

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


logger = logging.getLogger("xray_tools")

_ASC_MARKER = re.compile(r"ASC\d+")
_CONT_MARKER = re.compile(r"\bCONT\s*:")
_USERLIMIT_PATTERN = re.compile(
    r"userlimit=\((?P<column>\w+)>(?P<min>[-+0-9.eE]+)&&(?P=column)<(?P<max>[-+0-9.eE]+)\)"
)


@dataclass(frozen=True)
class LightCurveCuts:
    """Count-rate limits that a GTI file was built from."""

    rate_column: str
    min_rate: float
    max_rate: float


def _parse_userlimit_cuts(records: list[list[str]]) -> LightCurveCuts | None:
    """Recover count-rate limits from a GTI file's HISTORY records.

    CIAO's GTI tools record the rate filter as ``userlimit=(col>lo && col<hi)``
    in HISTORY cards, split across continuation records. Returns None when the
    records carry no rate limit.
    """
    text = " ".join(
        record[1] for record in records if len(record) >= 2 and record[0] == "HISTORY"
    )
    text = _ASC_MARKER.sub("", text)
    text = _CONT_MARKER.sub("", text)
    text = re.sub(r"\s+", "", text)
    match = _USERLIMIT_PATTERN.search(text)
    if match is None:
        return None
    return LightCurveCuts(
        rate_column=match.group("column"),
        min_rate=float(match.group("min")),
        max_rate=float(match.group("max")),
    )


@dataclass(frozen=True)
class CIAOGTI:
    """A reference to a file-backed CIAO good-time interval product."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def to_pandas(self) -> pd.DataFrame:
        """Copy the GTI START/STOP intervals into a pandas DataFrame."""
        import pandas as pd

        require_file(self.path, "GTI file")
        # GTI products open on their (empty) FILTER block, so the GTI
        # extension has to be selected explicitly.
        try:
            crate = require_pycrates().read_file(f"{self.path}[GTI]")
        except Exception:
            crate = require_pycrates().read_file(str(self.path))
        available = {name.casefold(): name for name in crate.get_colnames()}
        if not {"start", "stop"} <= set(available):
            raise ValueError("GTI file does not contain START and STOP columns")

        start = crate.get_column(available["start"]).values.copy()
        stop = crate.get_column(available["stop"]).values.copy()
        return pd.DataFrame({"start": start, "stop": stop, "duration": stop - start})

    @cached_property
    def cuts(self) -> LightCurveCuts | None:
        """Count-rate limits recorded when this GTI was created, if any."""
        require_file(self.path, "GTI file")
        try:
            crate = require_pycrates().read_file(f"{self.path}[FILTER]")
        except Exception as exc:  # a hand-made GTI may lack a FILTER block
            logger.debug("cannot read FILTER history of %s: %s", self.path, exc)
            return None
        return _parse_userlimit_cuts(crate.get_history_records())

    def plot(self, ax: Axes | None = None, *, title: str | None = None) -> Axes:
        """Draw each good-time interval as a horizontal bar."""
        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        intervals = self.to_pandas()
        time_offset = intervals["start"].iloc[0]
        ax.broken_barh(
            list(
                zip(
                    (intervals["start"] - time_offset) / 1e3,
                    intervals["duration"] / 1e3,
                )
            ),
            (0, 1),
        )
        ax.set_xlabel("Elapsed time (ks)")
        ax.set_ylabel("Good time")
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)
        return ax
