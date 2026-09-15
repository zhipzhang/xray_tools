"""Unit conversions frequently used in X-ray analysis."""

HC_KEV_ANGSTROM = 12.398419843320026
"""Planck's constant times the speed of light in keV Angstrom."""


def energy_to_wavelength(energy_kev: float) -> float:
    """Convert photon energy in keV to wavelength in Angstrom."""
    if energy_kev <= 0:
        raise ValueError("energy_kev must be positive")
    return HC_KEV_ANGSTROM / energy_kev


def wavelength_to_energy(wavelength_angstrom: float) -> float:
    """Convert photon wavelength in Angstrom to energy in keV."""
    if wavelength_angstrom <= 0:
        raise ValueError("wavelength_angstrom must be positive")
    return HC_KEV_ANGSTROM / wavelength_angstrom
