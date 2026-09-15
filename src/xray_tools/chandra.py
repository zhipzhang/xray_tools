"""Chandra instrument constants."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChandraACIS:
    """Reference values for Chandra's Advanced CCD Imaging Spectrometer.

    Values describe the detector configuration and should not replace values
    recorded in an individual observation's calibration products.
    """

    pixel_scale_arcsec: float = 0.492
    field_of_view_arcmin: float = 16.9
    energy_band_keV: tuple[float, float] = (0.2, 10.0)
