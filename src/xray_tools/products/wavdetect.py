"""File-backed source-detection products from CIAO's wavdetect tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import CIAOWorkflowError
from ..utils import (
    _column_components,
    prepare_output_path,
    require_ciao_runtool,
    require_file,
    require_pycrates,
)
from .image import CIAOImage

if TYPE_CHECKING:
    import pandas as pd


logger = logging.getLogger("xray_tools")


@dataclass(frozen=True)
class CIAOWavDetect:
    """File-backed wavdetect products for one detection run.

    ``sources`` is the FITS source list (SRCLIST block); ``cell`` maps image
    pixels to the source whose properties they were used to estimate;
    ``image`` is the reconstructed, de-noised source image; ``background`` is
    the normalized background estimated after source cleansing; ``regions``
    is the ds9 ASCII region file with the same sources as ``sources``.
    Construction is free; the source list is read through pycrates on first
    access.
    """

    sources: Path
    cell: CIAOImage | None = None
    image: CIAOImage | None = None
    background: CIAOImage | None = None
    regions: Path | None = None
    band: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", Path(self.sources))
        if self.regions is not None:
            object.__setattr__(self, "regions", Path(self.regions))
        for field in ("cell", "image", "background"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, CIAOImage):
                object.__setattr__(self, field, CIAOImage(value))

    def to_pandas(self) -> pd.DataFrame:
        """Copy the SRCLIST source table into a pandas DataFrame.

        Column names are lowercased. Virtual vector columns such as
        ``POS(X, Y)`` are expanded into their components (``x``, ``y``);
        true vector columns such as ``R`` become per-row lists, matching
        :meth:`CIAOEvents.to_pandas`.
        """
        import pandas as pd

        require_file(self.sources, "source list")
        crate = require_pycrates().read_file(f"{self.sources}[SRCLIST]")

        data = {}
        for name in crate.get_colnames():
            for component in _column_components(name):
                values = crate.get_column(component).values.copy()
                # pandas requires one-dimensional values per DataFrame column.
                data[component.lower()] = values if values.ndim == 1 else list(values)

        return pd.DataFrame(data)


def _wavdetect(
    infile: str | Path,
    outroot: str | Path,
    *,
    expfile: str | Path | None,
    psffile: str | Path | None,
    scales: str,
    sigthresh: float | None,
    falsesrc: float | None,
    ellsigma: float,
    expthresh: float,
    bkginput: str | Path | None,
    band: str | None,
    verbose: int,
    clobber: bool,
) -> CIAOWavDetect:
    """Run CIAO wavdetect and return its file-backed products.

    ``infile`` must be a counts image — wavdetect's probability distributions
    assume Poisson statistics and silently detect nothing on fluxed images.
    Outputs are derived from ``outroot`` with CIAO's autonaming suffixes:
    ``{outroot}_src.fits``, ``_scell.fits``, ``_image.fits``, ``_nbkg.fits``,
    and ``_src.reg``.
    """
    infile = Path(infile)
    require_file(infile, "counts image")
    for path, description in (
        (expfile, "exposure map"),
        (psffile, "PSF map"),
        (bkginput, "background image"),
    ):
        if path is not None:
            require_file(Path(path), description)

    raw = str(outroot)
    outputs = {
        "sources": Path(f"{raw}_src.fits"),
        "cell": Path(f"{raw}_scell.fits"),
        "image": Path(f"{raw}_image.fits"),
        "background": Path(f"{raw}_nbkg.fits"),
        "regions": Path(f"{raw}_src.reg"),
    }
    for index, path in enumerate(outputs.values()):
        prepare_output_path(path, clobber=clobber, create_parents=index == 0)

    options: dict[str, Any] = {}
    if expfile is not None:
        options["expfile"] = str(expfile)
    if psffile is not None:
        options["psffile"] = str(psffile)
    if sigthresh is not None:
        options["sigthresh"] = sigthresh
    if falsesrc is not None:
        options["falsesrc"] = falsesrc
    if bkginput is not None:
        options["bkginput"] = str(bkginput)

    runtool = require_ciao_runtool()
    logger.info("wavdetect: %s (scales=%s)", infile, scales)
    logger.info("outroot: %s", raw)
    runtool.wavdetect(
        infile=str(infile),
        outfile=str(outputs["sources"]),
        scellfile=str(outputs["cell"]),
        imagefile=str(outputs["image"]),
        defnbkgfile=str(outputs["background"]),
        regfile=str(outputs["regions"]),
        scales=scales,
        ellsigma=ellsigma,
        expthresh=expthresh,
        verbose=verbose,
        clobber=clobber,
        **options,
    )

    missing = [str(path) for path in outputs.values() if not path.is_file()]
    if missing:
        raise CIAOWorkflowError(f"wavdetect did not create: {', '.join(missing)}")
    return CIAOWavDetect(band=band, **outputs)
