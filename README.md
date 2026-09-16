# xray-tools

Pythonic, thin wrappers for reproducible Chandra/CIAO analysis. Event selections
remain lazy CIAO Data Model expressions; light curves, GTIs, and explicitly
written event selections are file-backed products.

## Installation

Install the package inside a configured CIAO 4.18 environment:

```bash
python -m pip install -e .
```

## Quick start

```python
from xray_tools import CIAOEvents

events = CIAOEvents("acisf03843_repro_evt2.fits")
background = events.energy(2500, 7000).exclude_region("source.reg")

print(background.dm)

lightcurve = background.lightcurve(
    "products/background.lc",
    binsize=259.28,
    clobber=True,
)
gti = lightcurve.clean("products/background_clean.gti", plot=False)

events.apply_gti(gti).write("products/clean_evt2.fits", clobber=True)
```

CIAO and `pycrates` are loaded only when an operation needs them, so the package
can still be imported in documentation and unit-test environments without CIAO.
