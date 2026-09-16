"""File-backed products created by CIAO workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .utils import prepare_output_path, require_file, require_lc_clean, require_pycrates

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


logger = logging.getLogger("xray_tools")


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
        )

        data = {}
        for name in wanted:
            actual = available.get(name.casefold())
            if actual is not None:
                data[name.lower()] = crate.get_column(actual).values.copy()

        return pd.DataFrame(data)

    def plot(self, ax: Axes | None = None) -> Axes:
        """Plot count rate against time on a Matplotlib axes."""
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

        ax.plot(data["time"], data[rate_column])
        ax.set_xlabel("TIME")
        ax.set_ylabel("Count rate")
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
        plot: bool = True,
        verbose: int = 1,
    ) -> CIAOGTI:
        """Create a GTI by applying CIAO's ``lc_clean`` algorithm."""
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
            plot=plot,
            verbose=verbose,
        )

        return CIAOGTI(output)


@dataclass(frozen=True)
class CIAOGTI:
    """A reference to a file-backed CIAO good-time interval product."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
