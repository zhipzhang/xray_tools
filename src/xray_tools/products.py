"""File-backed products created by CIAO workflows."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import prepare_output_path, require_file, require_lc_clean, require_pycrates

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


logger = logging.getLogger("xray_tools")

_ASC_MARKER = re.compile(r"ASC\d+")
_CONT_MARKER = re.compile(r"\bCONT\s*:")
_USERLIMIT_PATTERN = re.compile(
    r"userlimit=\((?P<column>\w+)>(?P<min>[-+0-9.eE]+)&&(?P=column)<(?P<max>[-+0-9.eE]+)\)"
)

_RATE_ERROR_COLUMNS = ("count_rate_err", "rate_err", "err_rate")


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
class CIAOLightCurve:
    """A reference to a file-backed CIAO light curve."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def to_pandas(self) -> pd.DataFrame:
        """Copy standard light-curve columns into a pandas DataFrame."""
        import pandas as pd

        require_file(self.path, "light-curve file")
        crate = require_pycrates().read_file(str(self.path))
        available = {name.casefold(): name for name in crate.get_colnames()}
        wanted = (
            "TIME",
            "TIME_MIN",
            "TIME_MAX",
            "COUNTS",
            "EXPOSURE",
            "COUNT_RATE",
            "RATE",
            "COUNT_RATE_ERR",
            "RATE_ERR",
            "ERR_RATE",
        )

        data = {}
        for name in wanted:
            actual = available.get(name.casefold())
            if actual is not None:
                data[name.lower()] = crate.get_column(actual).values.copy()

        return pd.DataFrame(data)

    def plot(
        self,
        ax: Axes | None = None,
        *,
        gti: CIAOGTI | str | Path | None = None,
        title: str | None = None,
    ) -> Axes:
        """Plot count rate against elapsed time, optionally with GTI overlay.

        Rates are drawn as a step function over the bin edges and, when the
        light curve carries an error column, with error bars at the bin
        centers. When ``gti`` is given, its good-time intervals are shaded
        across the panel, any count-rate cuts recorded in the GTI are drawn as
        dashed horizontal lines, and the total observation time against the
        GTI good time is annotated in the upper-left corner.
        """
        import numpy as np

        data = self.to_pandas()
        if "time" not in data:
            raise ValueError("light curve does not contain a TIME column")
        if "count_rate" in data:
            rate_column = "count_rate"
        elif "rate" in data:
            rate_column = "rate"
        else:
            raise ValueError("light curve does not contain COUNT_RATE or RATE")

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        rate = data[rate_column]
        if "time_min" in data:
            time_offset = data["time_min"].iloc[0]
        else:
            time_offset = data["time"].iloc[0]
        elapsed = (data["time"] - time_offset) / 1e3

        if {"time_min", "time_max"} <= set(data.columns):
            edges = np.concatenate(
                [data["time_min"].to_numpy(), data["time_max"].iloc[-1:].to_numpy()]
            )
            step_rate = np.concatenate([rate.to_numpy(), rate.iloc[-1:].to_numpy()])
            ax.plot(
                (edges - time_offset) / 1e3,
                step_rate,
                drawstyle="steps-post",
            )
            total_ks = (data["time_max"].iloc[-1] - time_offset) / 1e3
        else:
            ax.plot(elapsed, rate)
            total_ks = elapsed.iloc[-1]

        error_column = next((name for name in _RATE_ERROR_COLUMNS if name in data), None)
        if error_column is not None:
            ax.errorbar(elapsed, rate, yerr=data[error_column], fmt="none")

        ax.set_xlabel("Elapsed time (ks)")
        ax.set_ylabel("Count rate (count/s)")
        if title is not None:
            ax.set_title(title)

        if gti is not None:
            if not isinstance(gti, CIAOGTI):
                gti = CIAOGTI(gti)
            intervals = gti.to_pandas()
            for start, stop in zip(intervals["start"], intervals["stop"]):
                ax.axvspan(
                    (start - time_offset) / 1e3,
                    (stop - time_offset) / 1e3,
                    color="tab:green",
                    alpha=0.2,
                )
            good_ks = intervals["duration"].sum() / 1e3
            ax.text(
                0.02,
                0.98,
                f"Total observation: {total_ks:.2f} ks\n"
                f"GTI good time: {good_ks:.2f} ks ({good_ks / total_ks:.1%})",
                transform=ax.transAxes,
                va="top",
            )
            cuts = gti.cuts
            if cuts is not None:
                ax.axhline(cuts.min_rate, color="tab:red", linestyle="--", linewidth=1)
                ax.axhline(cuts.max_rate, color="tab:red", linestyle="--", linewidth=1)

        return ax

    def clean(
        self,
        outfile: str | Path,
        *,
        mean: float | None = None,
        clip: float = 3.0,
        sigma: float | None = None,
        scale: float = 1.2,
        minfrac: float = 0.1,
        verbose: int = 1,
    ) -> CIAOGTI:
        """Create a GTI by applying CIAO's ``lc_clean`` algorithm.

        ``lc_clean`` is always run with its own plotting disabled; visualize
        the result afterwards with :meth:`CIAOLightCurve.plot` and the
        returned GTI.
        """
        require_file(self.path, "light-curve file")
        # lc_clean itself always writes with clobber enabled.
        output = prepare_output_path(outfile, clobber=True)
        lc_clean = require_lc_clean()

        logger.info("lc_clean: %s", self.path)
        logger.info("output: %s", output)
        lc_clean(
            str(self.path),
            outfile=str(output),
            mean=mean,
            clip=clip,
            sigma=sigma,
            scale=scale,
            minfrac=minfrac,
            plot=False,
            verbose=verbose,
        )

        return CIAOGTI(output)


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
