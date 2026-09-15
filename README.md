# xray-tools

Small, dependency-light utilities and data models for Chandra and other X-ray
observatory data analysis.

## Installation

```bash
python -m pip install -e .
```

## Quick start

```python
from xray_tools import ChandraACIS, Observation, energy_to_wavelength

observation = Observation(
    observatory="Chandra",
    instrument="ACIS-I",
    obs_id="12345",
    exposure_ks=42.5,
)

print(observation.exposure_s)             # 42500.0
print(energy_to_wavelength(1.0))          # 12.398419843320026 Angstrom
print(ChandraACIS.pixel_scale_arcsec)     # 0.492
```

The package intentionally has no required runtime dependencies. It can be used
alongside Astropy, CIAO, and mission-specific analysis environments.
