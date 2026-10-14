# Design: `blanksky` support in xray_tools

Status: proposal, ready to implement from. Two preconditions already settled:
`CIAOImage` / `CIAOFluxImage` are implemented (see `image-design.md`), and runtool
exposure is **verified** — `runtool.blanksky`, `runtool.blanksky_image`, and
`runtool.blanksky_sample` are all registered in `ciao_contrib.runtool` (checked in
the local CIAO 4.18 environment), so the plain `runtool.blanksky(...)` call pattern
applies with no fallback needed.
References: [blanksky ahelp](https://cxc.cfa.harvard.edu/ciao/ahelp/blanksky.html)
(esp. `bkgparams`), [blanksky_image ahelp](https://cxc.cfa.harvard.edu/ciao/ahelp/blanksky_image.html).

## 1. What the CIAO tools do

`blanksky evtfile outfile asolfile [stowedbg] [weight_method] [bkgparams] [random] [clobber] [verbose]`

- Takes an **events file**, selects CalDB blank-sky background event files matching
  the observation (per CCD), combines and reprojects them to the observation's
  coordinates.
- Output is a **background events file** carrying header keywords:
  - `BKGMETH` — scaling method (`particle` | `time` | `particle-rate`)
  - `BKGSCALn` — per-chip scaling factors (`BKGSCALE` for single-chip/HRC)
- `weight_method` (default `particle`) selects the normalization; `bkgparams`
  (default: `[energy=9000:12000]` for ACIS, `[pi=300:500]` for HRC) is a DM filter
  of the form `[energy=lo:hi]` or `[pi=lo:hi]`, used only for `particle` /
  `particle-rate` and ignored for `time`.
- `asolfile` may be omitted (looked up from the `ASOLFILE` header keyword); accepts
  stack syntax (comma/space separated, or `@file.lis`).
- Caveat: blank-sky files are **not part of the standard CalDB install** — users
  must install them separately. Worth one docstring line.
- `stowedbg=True` (ACIS, CTI-corrected only) uses stowed particle background
  instead of blank-sky.

`blanksky_image bkgfile outroot imgfile [tmpdir] [clobber] [verbose]`

- Takes the tailored blanksky **events** file plus a **reference image**
  (concretely: the counts image produced by fluximage) and writes, depending on
  `BKGMETH`:
  - `{outroot}_particle_bgnd.img` + `{outroot}_particle_bkgsub.img`, or
  - `{outroot}_time_bgnd.img` + `{outroot}_time_bkgsub.img`
- The reference image defines the grid/binning and any energy filter applied to
  the background image.

## 2. Proposed API

```python
# events.py — on CIAOEvents (same shape as lightcurve() / fluximage())
evt.blanksky(
    outfile,
    *,
    asolfile=None,              # str | Path | sequence[str | Path]; None = ASOLFILE header
    stowedbg=False,
    weight_method="particle",   # "particle" | "time" | "particle-rate"
    bkgparams="default",        # "default" | "[energy=lo:hi]" | "[pi=lo:hi]"
    random=0,
    tmpdir=None,
    verbose=0,
    clobber=False,
) -> CIAOBlankSky

# products/blanksky.py — new classes (imports CIAOImage from products/image.py)
@dataclass(frozen=True)
class CIAOBlankSky:
    path: Path                      # tailored background events file

    @cached_property
    def method(self) -> str: ...              # BKGMETH keyword
    @cached_property
    def bkgparams(self) -> str | None: ...    # BKGPARAM filter actually used
    @cached_property
    def scales(self) -> dict[int, float]: ...  # BKGSCALn per chip (ACIS; {} for single-chip instruments)

    def images(
        self,
        outroot: str | Path,
        reference: str | Path | CIAOImage,   # counts map the background is scaled to
        *,
        tmpdir=None,
        verbose=0,
        clobber=False,
    ) -> CIAOBlankSkyImage: ...

@dataclass(frozen=True)
class CIAOBlankSkyImage:
    background: CIAOImage         # *_bgnd.img   — plot()/to_numpy() come free
    subtracted: CIAOImage         # *_bkgsub.img
```

Typical workflow:

```python
evt = CIAOEvents("evt.fits")
bkg = evt.blanksky("bkg.fits", weight_method="particle")
imgs = evt.fluximage("out/i", energy=[(0.5, 7)])
bg = bkg.images("out/b", imgs["0.5-7"].counts)   # CIAOImage; defines grid + energy band
bg.subtracted.plot()                              # background-subtracted, WCS axes
```

## 3. Does the counts-map dependency couple the parts too tightly?

No — and with `CIAOImage` implemented the coupling is lighter than originally
planned. Analysis against the current architecture:

- The package pattern is already: **CIAOEvents methods construct file-backed
  product objects; products may then run further CIAO tools on themselves.**
  Precedent: `CIAOEvents.lightcurve()` → `CIAOLightCurve`, whose `.clean()` calls
  `lc_clean` and returns a `CIAOGTI` (`products/lightcurve.py`). `CIAOBlankSky.images()`
  calling `blanksky_image` is exactly the same shape.
- `CIAOBlankSky` lives in its own `products/blanksky.py`, importing `CIAOImage`
  from `products/image.py` — a one-way edge, no events↔products cycle. (Moved
  out of image.py on 2026-09-24 so blanksky code stays independently
  navigable as it grows.)
- The reference is a plain parameter, `str | Path | CIAOImage`. There is no
  `CIAOFluxImage` case to special-case: since the CIAOImage refactor,
  `imgs["0.5-7"].counts` **is** a `CIAOImage`, so a FluxImage user just passes that.
  Normalization is `Path(getattr(reference, "path", reference))` — one line, no
  duck-typing matrix.
- What the reference gives us for free: `CIAOImage.energy_range` (ENERGYLO/HI) and
  grid metadata, so docstrings can state that the background image inherits the
  counts map's binning and band.

What to **avoid**: storing a `CIAOImage`/`CIAOFluxImage` as a field of
`CIAOBlankSky` ("embedding"), or having `fluximage()` trigger blanksky processing.
`CIAOBlankSky` keeps only its own events-file `path`; `CIAOBlankSkyImage` holding
two `CIAOImage`s is fine because those are themselves pure path-backed references.

Alternatives considered:

| Option | Verdict |
| --- | --- |
| A. `CIAOBlankSky.images(reference)` accepting `str \| Path \| CIAOImage` | **Recommended** — one normalization line, consistent with `clean()` / `apply_gti` precedents |
| B. Also accept `CIAOFluxImage` and duck-type `.counts` | Dropped — redundant since `.counts` is already a `CIAOImage`; keeps the type union minimal |
| C. `images()` accepts only a raw path; user passes `fluximg.counts.path` | Works, but accepting `CIAOImage` costs nothing and reads better |
| D. Put image generation on `CIAOEvents` (`evt.blanksky_image(bkg, ...)`) | Rejected: the subject of `blanksky_image` is the background file, not the source events; also forces events→products→events round trips |

## 4. Implementation steps

1. ~~Verify runtool exposure.~~ **Done (2026-09-19)**: `runtool.blanksky`,
   `runtool.blanksky_image`, `runtool.blanksky_sample` all present in
   `ciao_contrib.runtool` (CIAO 4.18).
2. **products/blanksky.py**
   - `CIAOBlankSky` dataclass with `path` field (`__post_init__` normalizes to
     `Path`, like the other products).
   - `method` / `bkgparams` / `scales` cached properties reading header keywords
     via `require_pycrates().read_file(self.path)` (same read pattern as
     `CIAOGTI.cuts`). The scaling keywords live on the EVENTS block, which is
     the block pycrates opens by default for event files — verified against a
     real blanksky output (ObsID 3843): `BKGMETH=particle`,
     `BKGPARAM=[energy=9000:12000]`, `BKGSCAL0/1/2/3/6/7 = 0.0394/0.0406/0.0370/
     0.0410/0.0358/0.1758` — one entry per CCD present in the observation.
     For `scales`: scan `get_keynames()` for `^BKGSCAL(\d+)$` → `{ccd: value}`.
     Single-chip instruments (HRC) write a plain `BKGSCALE` instead — scope is
     ACIS for now, so `scales` returns `{}` there rather than inventing a
     sentinel key. Follow-up when a real HRC file is at hand: a `scale` scalar
     property or `BKGSCALE` → `{0: ...}` mapping.
   - `images()` method: normalize reference to a path, `require_ciao_runtool()`,
     call `runtool.blanksky_image(bkgfile=..., outroot=..., imgfile=..., ...)`, then
     discover outputs with a `_discover_blanksky_images(outroot, since)` helper —
     glob for `{outroot}_*_{bgnd,bkgsub}.img` filtered by mtime ≥ run start,
     mirroring `_discover_flux_images`. Wrap both in `CIAOImage` and build
     `CIAOBlankSkyImage`.
   - Export both classes from `products/__init__.py` (and the top-level
     `xray_tools/__init__.py`).
3. **events.py**
   - `blanksky()` method: `require_file(self.filename)` → `prepare_output_path` →
     serialize `asolfile` (sequence → comma-joined stack) → `runtool.blanksky(...)`
     with only non-`None` optional kwargs (same conditional-options style as
     `fluximage()`) → construct `CIAOBlankSky(output)` via function-local import
     (avoids the events↔products cycle, as in `lightcurve()`).
   - Early validation: `weight_method` in `{"particle", "time", "particle-rate"}`;
     `bkgparams` matches `default` or `^\[(energy|pi)=\d+:\d+\]$` — raise
     `ValueError` before any CIAO call.
4. **Tests** (follow existing mock patterns, no CIAO needed):
   - `test_events.py`: patch `xray_tools.events.require_ciao_runtool`; assert exact
     call kwargs (asolfile stacking, bkgparams pass-through, clobber); parametrize
     validation errors; assert return type is `CIAOBlankSky` with the right path.
   - `test_products.py`: patch `xray_tools.products.blanksky.require_ciao_runtool` and
     `xray_tools.products.blanksky.require_pycrates` (fake header keywords for
     `method`/`scales`); tmp files + `os.utime` for discovery tests; test the
     reference passed as path vs `CIAOImage`.
5. **README**: add a short blanksky example to the quick start.

## 5. Scope notes and open questions

- **Instrument scope: ACIS first** (decided 2026-09-19). `scales` is ACIS-specific
  (`{}` on single-chip/HRC files, which write plain `BKGSCALE`). `method` /
  `bkgparams` work for any instrument, and `blanksky(..., bkgparams="[pi=300:500]")`
  accepts the HRC particle band — only the per-chip breakdown is ACIS-only, which
  is moot for a single-chip instrument. HRC `scales` support is a documented
  follow-up, not a blocker.
- **`to_pandas()` on `CIAOBlankSky`: not needed for now** (decided 2026-09-19) —
  the tailored file feeds `images()`, it is not an analysis table. If background
  spectra via `blanksky_sample` ever land, column access can be revisited then.
- **One-liner convenience** (`blanksky(..., counts=...)` running both steps)?
  Not recommended — the background events file is independently useful (e.g.
  background spectra via `blanksky_sample`, a possible future addition; runtool
  exposure already verified).
