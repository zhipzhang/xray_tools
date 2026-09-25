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
gti = lightcurve.clean("products/background_clean.gti")

lightcurve.plot(gti=gti, title="SS 433 background")  # shaded good times, cut levels, ontime
print(gti.cuts)           # count-rate limits lc_clean applied

events.apply_gti(gti).write("products/clean_evt2.fits", clobber=True)

images = events.fluximage(
    "products/ss433",      # trailing "/" would write into a directory instead
    energy=(0.5, 7.0),     # custom band; effective energy = sqrt(lo * hi)
    binsize=4,
    clobber=True,
)
images["0.5-7"].flux.plot()                    # log stretch, WCS axes, BUNIT colorbar
images["0.5-7"].counts.to_numpy(masked=True)   # MaskedArray; NaN pixels carry no data
images["0.5-7"].flux.path                      # products/ss433_0.5-7_flux.img

blank = events.blanksky("products/3843_blank_evt.fits", clobber=True)
blank.method, blank.scales          # "particle", {0: 0.039, 1: 0.041, ...} per chip
bgs = blank.images("products/ss433_bkg", images["0.5-7"].counts)
bgs.subtracted.plot()               # background-subtracted counts image

spec = events.region("src.reg").specextract(
    "products/3c273",               # writes products/3c273.{pi,arf,rmf}, 3c273_bkg.pi, ...
    bkgfile="bkg.reg",              # or a CIAOEvents / CIAOBlankSky background
    weight=False,                   # point source: unweighted responses ...
    correctpsf=True,                # ... with aperture-corrected ARF
    grouptype="NUM_CTS", binspec=15,
    clobber=True,
)
spec.counts, spec.exposure          # header metadata via pycrates
pha = spec.to_sherpa()              # sherpa DataPHA; ARF/RMF/background auto-attached
```

CIAO and `pycrates` are loaded only when an operation needs them, so the package
can still be imported in documentation and unit-test environments without CIAO.
