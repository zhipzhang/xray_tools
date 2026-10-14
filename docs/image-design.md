# Design: a generic `CIAOImage` wrapper

Status: implemented (2026-09-19); §7 records user decisions, §8 implementation
notes. Extends/supersedes the output-type parts of `docs/blanksky-design.md`
(the `CIAOBlankSkyImage` fields defined there become `CIAOImage` instances).

## 1. What the example outputs actually are

Probed `/disk/gamma/cta/work/zhipz/personal_data/Chandra_SS433/3843/analysis/flux`
(the `CIAOEvents.fluximage()` products for obsid 3843) with astropy + pycrates
(CIAO 4.18 env):

| file | HDU | shape | CONTENT | BUNIT (image unit) | zero frac | NaN frac |
| --- | --- | --- | --- | --- | --- | --- |
| `0.5-1.5_thresh.img` (counts) | PRIMARY 2D | (405, 485) | `EVT2` | — | 0.82 | 0 |
| `0.5-1.5_flux.img` | PRIMARY 2D | (405, 485) | `EVT2` | `/cm**2 /s` | 0.82 | 0 |
| `0.5-1.5_thresh.expmap` | PRIMARY 2D | (405, 485) | `EXP` | `cm**2 s` | 0.48 | 0 |
| `0.5-1.5_thresh.psfmap` | PRIMARY 2D | (405, 485) | `EVT2` | `arcsec` | 0 | 0.48 |
| `3843.fov` | **not an image** (FOV ext) | — | — | — | — | — |

So yes: every fluximage product except the FOV is "just a 2D image", and so are the
`blanksky_image` outputs (`*_bgnd.img`, `*_bkgsub.img`) from the blanksky design. One
wrapper class covers counts / flux / exposure / psf / background images. The FOV stays
a plain `Path` (pycrates reads it as a `TABLECrate`).

## 2. Does pycrates read images well? (verified empirically)

Yes — better than astropy for the data layer. Verified against the files above
(`ciao-4.18` env, `import pycrates as pc`):

```python
cr = pc.read_file(path)          # -> IMAGECrate (tables -> TABLECrate; use for validation)
pc.get_piximgvals(cr)            # -> np.ndarray float32, shape (ny, nx), NaNs PRESERVED
cr.get_shape()                   # -> (405, 485)
cr.get_image().get_unit()        # -> '/cm**2 /s'  (BUNIT lives on the image component)
cr.get_key_value("EXPOSURE")     # -> 67045.399...   (full 142-key header access)
cr.get_key_value("CONTENT")      # -> 'EVT2' / 'EXP'
cr.get_min_value() / cr.get_max_value()
cr.get_subspace_data(i, "energy")# -> energy range, WHEN present (see below)
cr.is_image()                    # -> True for images
```

Gotchas found while probing (document these in docstrings):

- **BUNIT is not visible via `cr.get_key_value("BUNIT")`** (returns `None`) — the unit
  lives on the image component: `cr.get_image().get_unit()`.
- NaNs are preserved as NaN (psfmap is 47.8% NaN off-chip) — stats helpers must use
  `np.nan*` functions.
- `cr.get_axis("sky")` returns a **flat** `(npix, 2)` `CrateData` of pixel-center
  physical coords — awkward for grids. Don't use it for WCS; use astropy.
- `WCSTransform` introspection is clunky/half-broken via SWIG. Don't build WCS on
  pycrates transforms.
- Combined products (`flux.img`, `.expmap`) carry **no energy subspace**
  (`get_subspace_data` raises) — energy range metadata must be optional/`None`.
  Binned counts images from events (`[bin sky=..][energy=..]`) do record it.

**Division of labor**: pycrates for data array + units + header keywords + subspace
(the package already standardizes on it via `require_pycrates()`); astropy for WCS,
plotting, and FITS interop (as `CIAOFluxImage.plot` already does).

## 3. Proposed `CIAOImage`

New class in `products/image.py` (same module as `CIAOFluxImage`; no new import edges):

```python
@dataclass(frozen=True)
class CIAOImage:
    path: Path            # file-backed reference only — no data loaded at construction

    # data
    def to_numpy(self, *, masked: bool = False) -> np.ndarray | np.ma.MaskedArray: ...
    #   default: raw float32 array as stored (NaNs preserved, no masking);
    #   masked=True: MaskedArray with NaN/inf masked — the "no data" marker
    #   (psfmap off-chip pixels). Reductions (mean/sum/min/...) skip masked
    #   elements automatically, unlike NaN which requires np.nan* everywhere;
    #   matplotlib renders masked pixels transparent.
    # metadata (cached_property, read via require_pycrates)
    shape: tuple[int, ...]
    unit: str | None                            # BUNIT from image component
    content: str | None                         # CONTENT keyword
    exposure: float | None                      # EXPOSURE keyword
    energy_range: tuple[float, float] | None    # ENERGYLO/ENERGYHI (keV), when recorded
    # visualization (moved from CIAOFluxImage.plot; data via pycrates, WCS via astropy)
    def plot(self, ax=None, *, stretch="log", cmap="inferno", smooth=None,
             title=None, vmin=None, vmax=None): ...
```

- Lazy like everything else in the package: construction is free, all reads happen in
  `to_numpy`/properties via the `require_pycrates()` seam (keeps tests mockable with no
  CIAO installed).
- Bonus capability, free from pycrates: `read_file` accepts DM filter strings
  (`"evt.fits[bin sky=32][energy=500:7000]"` reads as an image) — mention in the
  docstring, don't build API around it yet.

## 4. Refactoring `CIAOFluxImage` onto `CIAOImage`

Current fields (`products/image.py`): `counts/flux/exposure/fov/psf: Path | None`.
Proposal: change the image fields to `CIAOImage`, keep `fov: Path | None`:

```python
@dataclass(frozen=True)
class CIAOFluxImage:
    band: str
    counts: CIAOImage | None
    flux: CIAOImage
    exposure: CIAOImage | None
    fov: Path | None
    psf: CIAOImage | None

    def plot(self, ax=None, *, component="flux", **kwargs): ...
    # delegates to getattr(self, component).plot(ax=ax, **kwargs)
```

- Still file-backed (`CIAOImage` holds only a `Path`) — no lifecycle coupling.
- Attribute names unchanged, so call sites like `imgs["0.5-7"].flux.path` keep working
  modulo a `.path` where a raw path was expected. Package is pre-alpha (0.1.0); this
  breaking change is the right moment for it.
- `_discover_flux_images` (`products/image.py`) constructs `CIAOImage(path)` instead of
  keeping bare paths.
- Downstream impact on the blanksky design: `CIAOBlankSkyImage.background` /
  `.subtracted` become `CIAOImage`, and `CIAOBlankSky.images(reference=...)` accepts
  `str | Path | CIAOImage` (FluxImage users pass `.counts`, which is already a
  `CIAOImage`).

## 5. Implementation steps (next step)

1. **products/image.py**: add `CIAOImage` (data via `require_pycrates()` + `get_piximgvals`;
   metadata via `get_image().get_unit()` / `get_key_value` / ENERGYLO-ENERGYHI band keys;
   `plot` moved from `CIAOFluxImage.plot` with identical signature plus vmin/vmax).
2. **products/image.py**: switch `CIAOFluxImage` fields to `CIAOImage` (`fov` stays `Path`);
   make `plot(component=...)` delegate; update `_discover_flux_images`.
3. **__init__.py**: export `CIAOImage`.
4. **Tests** (existing mock patterns, no CIAO needed):
   - `test_products.py`: patch `xray_tools.products.image.require_pycrates`; fake
     `IMAGECrate` (shape/unit/keys/subspace + `get_piximgvals` returning an array
     seeded with NaN) to test `to_numpy` raw vs `masked` + metadata props, including
     the None paths (no BUNIT, no energy subspace, non-image file ->
     `CIAOWorkflowError`).
     Keep/adapt the existing real-FITS `plot` tests; assert delegation for each
     `component`.
   - Update discovery tests to assert `CIAOImage` fields.
   - `tests/test_core.py` untouched.
5. **README**: short example `evt.fluximage("out/i")["broad"].flux.plot()`.

## 6. Open questions

- Name: `CIAOImage` vs `CIAOImageProduct`? (Leaning `CIAOImage` — short, matches
  `CIAOEvents`/`CIAOGTI` style.)

## 7. Decided / deferred

- **`to_numpy(masked=False)`** (user decision, 2026-09-19) — `masked=True` returns a
  `np.ma.MaskedArray` with NaN/inf masked; default stays the raw array.
  `mask_zero` was considered and dropped the same day: zeros are overwhelmingly real
  measurements (zero counts on an exposed pixel; zero flux below `expmapthresh`),
  display already handles zeros via vmin/norm clipping, and excluding no-coverage
  pixels properly belongs to a future exposure-threshold mechanism, not a blind zero
  mask. Implementation: `np.ma.masked_invalid(data)`.

## 8. Implementation notes (2026-09-19)

- **`energy_range` reads `ENERGYLO`/`ENERGYHI` header keywords (keV)**, not the DM
  subspace: verified that fluximage strips the energy subspace from all its products
  (counts/flux/expmap alike — `get_subspace_data` finds no `energy` item), while
  `ENERGYLO`/`ENERGYHI` are present and reliable. DM subspace trivia if ever needed:
  components are 1-based; `CrateSubspaceData` exposes `range_min`/`range_max`
  properties, not `get_range()`.
- **The plot colorbar label comes from the FITS BUNIT keyword** (read via astropy in
  `plot`), replacing the old hardcoded per-component label table — data-driven and
  correct for arbitrary images (e.g. `arcsec` for psfmaps, `cm**2 s` for expmaps).
- `to_numpy`/`_crate` validate with `crate.is_image()` and raise `CIAOWorkflowError`
  for non-image files (the FOV file reads as a `TABLECrate`).
- Verified end-to-end on the SS433/3843 products: masked psfmap mean skips the 47.8%
  off-chip NaNs; `flux.plot()` renders with WCS axes and the `/cm**2 /s` colorbar.
- **plot lives on `CIAOImage`** (user decision, 2026-09-19) — `CIAOFluxImage.plot`
  becomes a thin delegate via `component=`. The vmin/vmax knobs in the §3 signature
  stay. Open to the implementation: use the masked array (non-positive values masked)
  internally for log stretch.
- **Image arithmetic (`__sub__`, `__truediv__` → `dmimgcalc`) is explicitly out of
  scope for now** — deferred by user until the `CIAOImage` design is familiar in
  practice. When picked up: separate small design; motivating use case is
  exposure-ratio scaling that `blanksky_image` does not precompute
  (`*_bkgsub.img` already covers plain subtraction).
