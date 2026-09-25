"""File-backed 2D image products from CIAO imaging workflows."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..exceptions import CIAOWorkflowError
from ..utils import require_file, require_pycrates

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes

    from .wavdetect import CIAOWavDetect


logger = logging.getLogger("xray_tools")


@dataclass(frozen=True)
class CIAOImage:
    """A reference to a file-backed 2D image product.

    Construction is free and needs no CIAO environment; pixel data and
    metadata are read through pycrates on first access. NaN marks pixels
    with no data (for example the off-chip regions of a PSF map) and is
    preserved as stored.
    """

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def _crate(self) -> Any:
        require_file(self.path, "image")
        crate = require_pycrates().read_file(str(self.path))
        if not crate.is_image():
            raise CIAOWorkflowError(f"not a 2D image: {self.path}")
        return crate

    def to_numpy(self, *, masked: bool = False) -> np.ndarray | np.ma.MaskedArray:
        """Return the pixel values as a float array, shaped ``(ny, nx)``.

        With ``masked=True`` the result is a masked array whose NaN/inf
        entries — the "no data" marker — are masked, so reductions skip
        them without needing ``np.nan*`` variants.
        """
        import numpy as np

        pycrates = require_pycrates()
        data = np.asarray(pycrates.get_piximgvals(self._crate()))
        if masked:
            return np.ma.masked_invalid(data)
        return data

    @cached_property
    def shape(self) -> tuple[int, ...]:
        """Image dimensions as ``(ny, nx)``."""
        return tuple(int(axis) for axis in self._crate().get_shape())

    @cached_property
    def unit(self) -> str | None:
        """The BUNIT of the image component, e.g. ``"/cm**2 /s"``."""
        unit = self._crate().get_image().get_unit()
        return unit or None

    @cached_property
    def content(self) -> str | None:
        """The CIAO CONTENT keyword, e.g. ``"EVT2"`` or ``"EXP"``."""
        value = self._crate().get_key_value("CONTENT")
        return value or None

    @cached_property
    def exposure(self) -> float | None:
        """The EXPOSURE keyword in seconds, when recorded."""
        value = self._crate().get_key_value("EXPOSURE")
        return float(value) if value is not None else None

    @cached_property
    def energy_range(self) -> tuple[float, float] | None:
        """Band edges in keV from the ENERGYLO/ENERGYHI keywords, when recorded."""
        crate = self._crate()
        lo = crate.get_key_value("ENERGYLO")
        hi = crate.get_key_value("ENERGYHI")
        if lo is None or hi is None:
            return None
        return (float(lo), float(hi))

    def plot(
        self,
        ax: Axes | None = None,
        *,
        stretch: str = "log",
        cmap: str = "inferno",
        smooth: float | None = None,
        title: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> Axes:
        """Display the image with Matplotlib.

        The display stretch is ``"log"``, ``"sqrt"``, or ``"linear"``;
        non-positive values are masked for the logarithmic stretch. ``smooth``
        gives the sigma of a Gaussian kernel, in image pixels, applied for
        display only. ``vmin``/``vmax`` override the data range used for the
        stretch. When this method creates the axes itself, the image's
        celestial WCS is used to label the axes with RA and Dec, and the
        colorbar is labelled from the BUNIT keyword when the file records one.
        """
        import numpy as np
        from astropy.io import fits
        from astropy.visualization import (
            ImageNormalize,
            LinearStretch,
            LogStretch,
            SqrtStretch,
        )

        stretches = {"linear": LinearStretch, "sqrt": SqrtStretch, "log": LogStretch}
        if stretch not in stretches:
            raise ValueError(f"stretch must be one of {sorted(stretches)}")
        if smooth is not None and smooth <= 0:
            raise ValueError("smooth must be positive")

        require_file(self.path, "image")
        data = np.asarray(self.to_numpy(), dtype=float)
        with fits.open(self.path) as hdul:
            header = hdul[0].header.copy()

        data = np.where(np.isfinite(data), data, np.nan)
        if smooth is not None:
            from astropy.convolution import Gaussian2DKernel, convolve

            data = convolve(
                data,
                Gaussian2DKernel(smooth),
                normalize_kernel=True,
                boundary="extend",
            )
        if stretch == "log":
            data = np.where(data > 0, data, np.nan)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            raise ValueError("the image has no displayable data")

        if ax is None:
            import matplotlib.pyplot as plt

            projection = None
            try:
                from astropy.wcs import WCS

                wcs = WCS(header)
                if wcs.has_celestial:
                    projection = wcs
            except Exception:
                projection = None
            if projection is None:
                _, ax = plt.subplots()
                ax.set_xlabel("Sky X (pixel)")
                ax.set_ylabel("Sky Y (pixel)")
            else:
                _, ax = plt.subplots(subplot_kw={"projection": projection})

        norm = ImageNormalize(
            vmin=finite.min() if vmin is None else vmin,
            vmax=finite.max() if vmax is None else vmax,
            stretch=stretches[stretch](),
            clip=False,
        )
        image = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
        ax.figure.colorbar(image, ax=ax, label=header.get("BUNIT"))
        if title is not None:
            ax.set_title(title)
        return ax


@dataclass(frozen=True)
class CIAOFluxImage:
    """Exposure-corrected image products for one energy band."""

    band: str
    counts: CIAOImage | None
    flux: CIAOImage
    exposure: CIAOImage | None
    fov: Path | None
    psf: CIAOImage | None

    def __post_init__(self) -> None:
        for field in ("counts", "flux", "exposure", "psf"):
            value = getattr(self, field)
            if value is not None and not isinstance(value, CIAOImage):
                object.__setattr__(self, field, CIAOImage(value))
        if self.fov is not None:
            object.__setattr__(self, "fov", Path(self.fov))

    def plot(
        self,
        ax: Axes | None = None,
        *,
        component: str = "flux",
        **kwargs,
    ) -> Axes:
        """Display one of the band's images with Matplotlib.

        ``component`` selects which image is shown: ``"flux"`` (the default),
        ``"counts"``, ``"exposure"``, or ``"psf"``. Additional keyword
        arguments are forwarded to :meth:`CIAOImage.plot`.
        """
        components = {
            "flux": self.flux,
            "counts": self.counts,
            "exposure": self.exposure,
            "psf": self.psf,
        }
        if component not in components:
            raise ValueError(f"unknown image component: {component!r}")
        image = components[component]
        if image is None:
            raise ValueError(f"the {component} image is not available for this band")
        return image.plot(ax=ax, **kwargs)

    def wavdetect(
        self,
        outroot: str | Path,
        *,
        scales: Sequence[float] = (1, 2, 4, 8, 16),
        sigthresh: float | None = None,
        falsesrc: float | None = None,
        ellsigma: float = 3.0,
        expthresh: float = 0.1,
        background: CIAOImage | str | Path | None = None,
        verbose: int = 0,
        clobber: bool = False,
    ) -> CIAOWavDetect:
        """Detect sources in this band's counts image with CIAO's ``wavdetect``.

        The exposure map and PSF map components are wired up automatically,
        so the three inputs always share the grid of the fluximage run that
        produced them. wavdetect assumes Poisson statistics, so the detection
        always runs on the ``counts`` component, never on the fluxed image.

        ``scales`` lists the Mexican-Hat wavelet radii in image pixels; the
        default ``(1, 2, 4, 8, 16)`` follows the wavdetect thread's
        recommendation for Chandra data and deliberately diverges from CIAO's
        minimal ``"2.0 4.0"`` default. ``sigthresh`` is the per-pixel
        detection significance (CIAO default 1e-6, ~1 false source per
        1000x1000 pixels); ``falsesrc`` instead sets a per-scale false-source
        budget and, when given, makes CIAO ignore ``sigthresh``. ``ellsigma``
        scales the source ellipses written to the region file; ``expthresh``
        drops pixels whose relative exposure falls below the given fraction.
        ``background`` supplies a predetermined normalized background image
        (wavdetect's ``bkginput``) instead of estimating one from the data.

        ``outroot`` is a file prefix: ``"detect/3843_2-7"`` writes
        ``detect/3843_2-7_src.fits`` and its ``_scell.fits`` / ``_image.fits``
        / ``_nbkg.fits`` / ``_src.reg`` siblings, creating missing
        directories. The ``regions`` product feeds straight back into
        :meth:`CIAOEvents.region` / :meth:`CIAOEvents.exclude_region`.
        """
        from .wavdetect import _wavdetect

        if self.counts is None:
            raise CIAOWorkflowError(
                f"wavdetect needs the counts image of the band; "
                f"the {self.band!r} band has none"
            )
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive wavelet radii in pixels")
        if sigthresh is not None and sigthresh <= 0:
            raise ValueError("sigthresh must be positive")
        if falsesrc is not None and falsesrc <= 0:
            raise ValueError("falsesrc must be positive")
        if ellsigma <= 0:
            raise ValueError("ellsigma must be positive")
        if not 0 < expthresh <= 1:
            raise ValueError("expthresh must be within (0, 1]")

        raw = str(outroot)
        if raw.endswith("/"):
            raise ValueError(
                "outroot is a file prefix, not a directory: "
                f'use "{raw}name" to write into a directory'
            )

        background_path = None
        if background is not None:
            background_path = (
                background.path if isinstance(background, CIAOImage) else Path(background)
            )
            require_file(background_path, "background image")

        if self.psf is None:
            logger.warning(
                "no PSF map for the %r band: detection is unaffected, but source "
                "characteristics are unreliable; run fluximage(psfecf=...)",
                self.band,
            )
        if falsesrc is not None and sigthresh is not None:
            logger.info("falsesrc is set; wavdetect ignores sigthresh")

        npix = 1
        for axis in self.counts.shape:
            npix *= axis
        if npix > 2048 * 2048:
            logger.warning(
                "image has %d pixels; wavdetect needs at least 1-2 GB of memory "
                "beyond 2048x2048",
                npix,
            )

        return _wavdetect(
            self.counts.path,
            raw,
            expfile=self.exposure.path if self.exposure is not None else None,
            psffile=self.psf.path if self.psf is not None else None,
            scales=" ".join(f"{scale:g}" for scale in scales),
            sigthresh=sigthresh,
            falsesrc=falsesrc,
            ellsigma=ellsigma,
            expthresh=expthresh,
            bkginput=background_path,
            band=self.band,
            verbose=verbose,
            clobber=clobber,
        )


def _discover_flux_images(
    outroot: str | Path, *, since: float
) -> dict[str, CIAOFluxImage]:
    """Map band labels to the fluximage products created after ``since``.

    Only files newer than the given timestamp are considered, so products
    from earlier runs with the same outroot are left out.
    """
    raw = str(outroot)
    directory_style = raw.endswith("/")
    base = Path(raw) if directory_style else Path(raw).parent
    prefix = "" if directory_style else Path(raw).name

    def fresh(path: Path) -> bool:
        return path.is_file() and path.stat().st_mtime >= since - 1.0

    def first_fresh(*names: str) -> Path | None:
        for name in names:
            candidate = base / name
            if fresh(candidate):
                return candidate
        return None

    def label_of(flux_name: str) -> str:
        stem = flux_name[: -len("_flux.img")]
        return stem.removeprefix(f"{prefix}_") if prefix else stem

    flux_pattern = "*_flux.img" if not prefix else f"{prefix}_*_flux.img"
    fov_pattern = "*.fov" if not prefix else f"{prefix}_*.fov"

    images: dict[str, CIAOFluxImage] = {}
    for flux in sorted(base.glob(flux_pattern)):
        if not fresh(flux):
            continue
        stem = flux.name[: -len("_flux.img")]
        fovs = [path for path in sorted(base.glob(fov_pattern)) if fresh(path)]
        band = label_of(flux.name)
        images[band] = CIAOFluxImage(
            band=band,
            counts=first_fresh(f"{stem}_thresh.img", f"{stem}.img"),
            flux=flux,
            exposure=first_fresh(f"{stem}_thresh.expmap", f"{stem}.expmap"),
            fov=fovs[0] if fovs else None,
            psf=first_fresh(f"{stem}_thresh.psfmap", f"{stem}.psfmap"),
        )
    return images
