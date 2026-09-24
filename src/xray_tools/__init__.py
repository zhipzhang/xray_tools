"""Utilities and data models for X-ray observatory analysis."""

from .chandra import ChandraACIS
from .events import CIAOEvents, merge_events
from .exceptions import CIAOWorkflowError
from .observation import Observation
from .products import (
    CIAOGTI,
    CIAOBlankSky,
    CIAOBlankSkyImage,
    CIAOFluxImage,
    CIAOImage,
    CIAOLightCurve,
    LightCurveCuts,
)
from .units import energy_to_wavelength, wavelength_to_energy

__all__ = [
    "CIAOGTI",
    "CIAOBlankSky",
    "CIAOBlankSkyImage",
    "CIAOEvents",
    "CIAOFluxImage",
    "CIAOImage",
    "CIAOLightCurve",
    "CIAOWorkflowError",
    "ChandraACIS",
    "LightCurveCuts",
    "Observation",
    "energy_to_wavelength",
    "merge_events",
    "wavelength_to_energy",
]

__version__ = "0.1.0"
