# Design: wavelet source detection via `wavdetect`, bound to `CIAOFluxImage`

Status: implemented (2026-09-25); §7 records implementation notes and the
deviations found during real-data verification.
References: [wavdetect ahelp](https://cxc.cfa.harvard.edu/ciao/ahelp/wavdetect.html),
[wavdetect thread](https://cxc.cfa.harvard.edu/ciao/threads/wavdetect/index.html)
(CIAO 4.18), Freeman et al. 2002, ApJS, 138, 185.
Precondition verified (2026-09-25): `runtool.wavdetect` is registered in
`ciao_contrib.runtool` (CIAO 4.18).
Binding to `CIAOFluxImage` is the user's idea (user decision, 2026-09-25).

## 1. What `wavdetect` does, and what we must respect

`wavdetect` is a CIAO wrapper over two tools:

- **`wtransform`** — correlates the input image with Mexican-Hat wavelets at
  each user scale, iteratively "cleansing" pixels with significant positive
  correlation so the background can be estimated from the cleansed image
  (`maxiter`/`iterstop`/`bkgsigthresh` control this). Per scale it writes a
  correlation-maxima table, correlation image, and normalized background.
- **`wrecon`** — reconciles the per-scale maxima into one source list
  (deciding which maxima at which scales are the same source), computes a
  source cell containing most of the flux, and derives source properties
  (position, counts, rates, shape, PSF size) within that cell.

Facts that shape the wrapper:

- **The input must be a counts image.** The probability distributions assume
  Poisson statistics; on a fluxed image the correlations are so small that
  *nothing* is detected (ahelp `infile`). This is the single most important
  correctness constraint — and the reason `CIAOFluxImage` is the right home:
  its `counts` component is exactly the `*_thresh.img` counts image.
- **Congruent inputs for free.** wavdetect takes three images that must share
  a grid: `infile` (counts), `expfile` (exposure map, optional but suppresses
  false positives near exposure edges), `psffile` (PSF-size map from
  `mkpsfmap`, optional; without it detection still works but source
  characteristics — `PSF_SIZE` etc. — are unreliable). `CIAOFluxImage` already
  bundles `counts` + `exposure` + `psf` from one `fluximage` run (the psfmap
  is written automatically when `fluximage(psfecf=...)` is set, CIAO ≥ 4.11),
  so congruence is guaranteed by construction.
- **Outputs are four required files plus an optional region file**, with CIAO
  autonaming suffixes (ahelp): `outfile` → `_src` (FITS source list, SRCLIST
  block), `scellfile` → `_scell` (pixels used per source), `imagefile` →
  `_image` (reconstructed, de-noised source image), `defnbkgfile` → `_nbkg`
  (normalized background), `regfile` (ASCII ds9 regions, same sources as
  `outfile`).
- **Detection tool, not photometry.** Net counts are not corrected for
  vignetting, QE, dead area, pileup, or PSF fraction; positions errors are
  counts-weighted centroid variances. Fine for detection + quick look;
  downstream `specextract()` does the real measurement.
- **`sigthresh` scales with image size**: ~1/Npixels keeps ~1 false source per
  field (default 1e-6 ≈ one spurious source in a 1000×1000 map). `falsesrc ≥
  0` switches to a per-scale false-source budget calibrated from simulations
  (`sigcalfile`), ignoring `sigthresh`.
- **Memory**: 512×512 → ~36 MB, 2048×2048 → >300 MB; larger images need ≥1–2 GB
  and page heavily if they do not fit.

Gotchas found in the ahelp worth documenting, not fixing:

- `PSF_SIZE`/`PSFRATIO` may be NaN for otherwise valid sources.
- `R` can be 0 for delta-function sources (a small blocking-based value is
  substituted for display).
- Near-duplicate sources in the output are a known condition; CIAO ≥ 4.12 adds
  diagnostic columns (`CORRELATION_MAX`, `FLUX_MAX`, `SCALES`) to spot them.
- Parallel instances collide on temp stack files unless `ASCDS_WORK_PATH`
  differs per run — an environment concern, out of the wrapper's scope.
- The source-cell image only represents the *last* (largest) scale; per-scale
  overlap is intrinsic to the wavelet decomposition.

## 2. Parameter triage: what the wrapper exposes

wavdetect has ~26 parameters. Same premise as specextract-design.md §2: expose
the core set as explicit kwargs, leave the rest at CIAO defaults, no
`**kwargs` passthrough. `infile`/`expfile`/`psffile` come from `self`.

### Core (explicit kwargs)

| wavdetect param | proposed kwarg | default | why core |
| --- | --- | --- | --- |
| `infile` | `self.counts` | — | the Poisson counts image; error when `None` |
| `expfile` | `self.exposure` | — | exposure map from the same fluximage run; passed only when present |
| `psffile` | `self.psf` | — | PSF sizes; warn when `None` (detection unaffected, characterization unreliable) |
| `outfile`/`scellfile`/`imagefile`/`defnbkgfile`/`regfile` | derived from `outroot` (positional) | — | one required argument, CIAO autonaming suffixes (§3) |
| `scales` | `scales` | `(1, 2, 4, 8, 16)` | the main science knob; serialized to `"1 2 4 8 16"` form (§4) |
| `sigthresh` | `sigthresh` | `None` → CIAO 1e-6 | detection significance threshold |
| `falsesrc` | `falsesrc` | `None` → CIAO -1 | per-scale false-source budget; ≥0 overrides `sigthresh` |
| `ellsigma` | `ellsigma` | `3.0` | region ellipse size for ds9 overlay |
| `expthresh` | `expthresh` | `0.1` | relative-exposure cutoff; trims chip edges |
| `bkginput` | `background` | `None` | predetermined normalized background image (`CIAOImage \| str \| Path`) — the future blanksky-image integration point |
| `verbose` | `verbose` | `0` | housekeeping |
| `clobber` | `clobber` | `False` | housekeeping |

### Excluded (stay at CIAO defaults; add later only on demand)

| wavdetect param | default | why excluded |
| --- | --- | --- |
| `bkgsigthresh` | 0.001 | cleansing-iteration internal; must stay ≥ `sigthresh`, easy to get wrong |
| `maxiter` | 2 | cleansing-iteration internal; tool self-stops via `iterstop` |
| `iterstop` | 1e-4 | cleansing-iteration internal |
| `exptime` | 0 | with an expfile, 0 means "use the expmap value at the source" — the right behavior; our expmap comes from fluximage |
| `bkgtime` | 0 | only meaningful with `bkginput`; revisit when blanksky backgrounds land |
| `bkgerrinput` | no | ditto (background error block) |
| `sigcalfile` | `$ASCDS_CALIB/wtsimresult.fits` | calibration ARD shipped with CIAO; overriding is expert territory |
| `interdir` | `.` | temp-file plumbing |
| `outputinfix` | — | name-collision avoidance; our derived `outroot` names already namespace |
| `log` | no | runtool streams to stdout/Python logging |
| `regfile` as a param | — | always written as `{outroot}_src.reg`, never user-chosen |

## 3. Proposed API

```python
# products/image.py — on CIAOFluxImage (the pixels live here)
@dataclass(frozen=True)
class CIAOFluxImage:
    ...
    def wavdetect(
        self,
        outroot: str | Path,
        *,
        scales: Sequence[float] = (1, 2, 4, 8, 16),
        sigthresh: float | None = None,          # None -> CIAO 1e-6
        falsesrc: float | None = None,           # None -> CIAO -1 (sigthresh rules)
        ellsigma: float = 3.0,
        expthresh: float = 0.1,
        background: CIAOImage | str | Path | None = None,  # -> bkginput
        verbose: int = 0,
        clobber: bool = False,
    ) -> CIAOWavDetect: ...
```

```python
# products/wavdetect.py — new file-backed product bundle
@dataclass(frozen=True)
class CIAOWavDetect:
    sources: Path        # {outroot}_src.fits   (SRCLIST table)
    cell: CIAOImage      # {outroot}_scell.fits (source-cell map)
    image: CIAOImage     # {outroot}_image.fits (reconstructed source image)
    background: CIAOImage  # {outroot}_nbkg.fits (normalized background)
    regions: Path | None # {outroot}_src.reg    (ds9 ASCII ellipses)
    band: str | None     # carried from CIAOFluxImage.band

    def to_pandas(self) -> pd.DataFrame:
        """Read the SRCLIST table into a DataFrame (lowercase columns,
        vector columns as lists — same conventions as CIAOEvents.to_pandas)."""
```

Typical workflow:

```python
evt = CIAOEvents("acisf002254_repro_evt2.fits").ccd_id(0, 1, 2, 3)
imgs = evt.fluximage("out/2254", bands="broad", binsize=1, psfecf=0.393,
                     clobber=True)
det = imgs["broad"].wavdetect("out/2254_broad", sigthresh=4e-6)
src = det.to_pandas()[["ra", "dec", "net_counts", "psf_size"]]
det.image.plot()                       # reconstructed, de-noised image
masked = evt.exclude_region(det.regions)  # regions feed straight back into DM filters
```

## 4. Design decisions

- **Bound to `CIAOFluxImage`, not `CIAOEvents` or a free function** (user
  decision, 2026-09-25). wavdetect's three inputs are exactly the fluximage
  products bundled in `CIAOFluxImage`; binding there makes congruence the
  caller's non-problem and mirrors the dmimgcalc rule — methods belong where
  the pixels are referenced. The private engine is file-path based (§5), so a
  later `CIAOImage.wavdetect(expfile=..., psffile=...)` for hand-made or
  merged counts images stays a thin addition (§6).
- **`scales` defaults to `(1, 2, 4, 8, 16)`, diverging from the tool** (user
  decision, 2026-09-25). CIAO's `"2.0 4.0"` is documented as "fine for
  experimentation"; the thread recommends `"1 2 4 8 16"` for Chandra
  completeness, and its own example shows the default merging three core
  sources into one. The package's "defaults kept" principle
  (specextract-design.md §5) yields here because detection completeness is the
  tool's whole purpose; the docstring states the divergence.
- **Counts, never flux.** `self.counts` is the input; a `None` counts
  component is a `CIAOWorkflowError`, not a silent fall-back to `self.flux`
  (which would run cleanly and detect nothing — the worst failure mode).
  Document the Poisson requirement in the docstring.
- **One `outroot`, derived names.** wavdetect has no `outroot` concept and
  requires four output paths; the wrapper takes a single prefix and derives
  `{outroot}_src.fits`, `_scell.fits`, `_image.fits`, `_nbkg.fits`, `_src.reg`
  (CIAO's own autonaming suffixes). Trailing-`/` outroots are rejected with
  the specextract error message ("file prefix, not a directory"). Outputs are
  deterministic, so no mtime-filtered discovery glob is needed — construct the
  paths and `require_file` each after the run, raising `CIAOWorkflowError`
  listing any the tool failed to write.
- **Run the tool, never reimplement.** wtransform's per-scale cleansing
  iterations and the calibrated PSD thresholds (and the `falsesrc` simulation
  calibration) are exactly what we get for free by calling
  `runtool.wavdetect`; no new dependency (no scipy/photutils) enters the
  package.
- **`expfile`/`psffile` are auto-wired from the components, with no override
  kwargs** (user decision, 2026-09-25). Congruent-by-construction inputs are
  the point of binding to `CIAOFluxImage`; substitution or disable hatches
  wait for a workflow that needs them. psf `None` → warn and run: detection is
  unaffected without a psfmap, only characterization degrades (ahelp). Log a
  warning pointing at `fluximage(psfecf=...)`, pass no `psffile`.
- **`falsesrc` and `sigthresh` both exposed.** When `falsesrc >= 0`, CIAO
  ignores `sigthresh`; the wrapper forwards both verbatim and logs which one
  is in effect rather than inventing mutual-exclusion rules CIAO doesn't have.
- **`background` normalizes like `bkgfile` in specextract**: `CIAOImage` →
  `.path`, else `Path`; `require_file`. It maps to `bkginput`; `bkgtime`/
  `bkgerrinput` stay at CIAO defaults in v1.
- **Memory guard is a warning, not an error**: when `self.counts.shape`
  exceeds 2048×2048, log the ≥1–2 GB memory note before invoking.
- **Reading the source list**: `to_pandas()` reads
  `f"{sources}[SRCLIST]"` explicitly (the table is block 2; pycrates usually
  auto-selects it, but being explicit matches the ds9 `[SRCLIST]` convention
  and removes ambiguity). Column handling reuses the `_column_components`
  vector-expansion approach from `CIAOEvents.to_pandas` so `R`, `SCALES` etc.
  arrive sensibly.
- **`band` is carried onto the product** so multi-band detection results stay
  distinguishable (`imgs["broad"].wavdetect(...).band == "broad"`).

## 5. Implementation steps

1. Verify `runtool.wavdetect` is registered in `ciao_contrib.runtool`
   (CIAO 4.18) — one-line check, as for blanksky/specextract.
2. `products/wavdetect.py` (new file, so detection code stays independently
   navigable — the blanksky precedent):
   - `CIAOWavDetect` frozen dataclass; `__post_init__` normalizes paths and
     wraps image components in `CIAOImage` (the `CIAOFluxImage.__post_init__`
     pattern).
   - `to_pandas()` via `require_pycrates()`, explicit `[SRCLIST]` block.
   - `_wavdetect(infile, outroot-derived paths, *, expfile, psffile, scales,
     sigthresh, falsesrc, ellsigma, expthresh, bkginput, verbose, clobber)`
     private engine: `require_file` inputs, `prepare_output_path` on the
     `_src.fits` name with `create_parents=True`, conditional-options dict
     (only non-`None` forwarded), `runtool.wavdetect(...)`, `require_file`
     every expected output, return the `CIAOWavDetect`.
3. `products/image.py`: `CIAOFluxImage.wavdetect()` — validate (`counts`
   present, all `scales > 0`, `sigthresh`/`falsesrc` positive when set,
   outroot not directory-style), serialize scales as
   `" ".join(f"{s:g}" for s in scales)`, psf-None and >2048² warnings,
   function-local import of `products.wavdetect` (avoids the cycle, as in
   `events.py`), delegate to `_wavdetect` with `band=self.band`.
4. Exports: `CIAOWavDetect` into `products/__init__.py` and top-level
   `__init__.py` `__all__`.
5. Tests (existing mock seams, no CIAO):
   - `test_products.py`: patched
     `xray_tools.products.image.require_ciao_runtool` — assert exact runtool
     kwargs (infile = counts path, expfile/psffile wired from components,
     derived output names, default scales serialized as `"1 2 4 8 16"`,
     `bkginput` mapping); counts `None` → error; bad scales / trailing-`/`
     outroot / existing output without clobber → `ValueError`/`FileExistsError`;
     missing expected output after the run → `CIAOWorkflowError`;
     `CIAOWavDetect` path normalization.
   - `to_pandas()` via a fake TABLECrate (the `fake_pycrates` pattern),
     including a vector column.
6. README: short quick-start block as in §3, after the fluximage example.
7. Verify on real data against the command line: run
   `imgs["broad"].wavdetect(...)` and CLI `wavdetect` with identical inputs on
   the SS433 (or thread ObsID 2254) fluximage products; same source count,
   spot-check `RA`/`DEC`/`NET_COUNTS`/`PSF_SIZE`, and confirm
   `det.regions` loads into `CIAOEvents.exclude_region()`.

## 6. Open questions

- **`CIAOImage.wavdetect()` for non-fluximage counts images** (merged mosaics,
  hand-binned images): the engine already supports it; the public method is
  deferred until cross-observation image work exists (cf. dmimgcalc-design.md
  §5 reprojection note).
- **`plot()` overlay** (reconstructed image + ellipses from the source list)?
  Nice quick-look; defer — matplotlib is an optional dependency and the
  regfile already covers ds9.
- **Structured region objects from the source list** (typed ellipse/region
  values instead of raw paths/DataFrames): no existing package code to build
  on; revisit only when a workflow needs programmatic regions.

## 7. Implementation notes (2026-09-25)

- All steps in §5 landed; 13 new tests in `tests/test_products.py`, full suite
  green (128 passed).
- **Test seam correction**: the runtool mock patches
  `xray_tools.products.wavdetect.require_ciao_runtool` (the engine's
  namespace), not `products.image` as §5's draft said — the standard
  "patch the name in the using module" rule. The method-level memory warning
  additionally needs `xray_tools.products.image.require_pycrates` patched,
  since it reads `self.counts.shape`.
- **`to_pandas()` virtual vector columns** — found during real-data
  verification: `get_colnames()` on a SRCLIST table returns labels like
  `POS(X, Y)` and `SCALES(X_WAVELET_SCALE, Y_WAVELET_SCALE)` that
  `get_column()` cannot fetch; the components (`X`, `Y`, ...) are the real
  columns. `_column_components` moved from `events.py` to `utils.py` (shared),
  and `to_pandas()` expands those labels into per-component columns, so the
  frame has `x`/`y`/`x_wavelet_scale`/... plus true vector columns (`r`) as
  per-row lists.
- **`prepare_output_path` checks all five outputs**, not just `_src.fits`, so
  a stale sibling (e.g. an old `_nbkg.fits`) also triggers `FileExistsError`
  before CIAO runs.
- **Verified end-to-end on real data** (SS 433, ObsID 3843, 2–7 keV band,
  405×485 counts image): wrapper run and CLI `wavdetect` with identical
  inputs produced **numerically identical** source lists (54 sources;
  RA/DEC/NET_COUNTS/PSF_SIZE/SRC_SIGNIFICANCE compared with
  `allclose(rtol=0, atol=1e-10)`). The `regions` output round-trips through
  `CIAOEvents.exclude_region()`: 333825 → 321599 events after masking the
  detected sources. Products left in
  `.../Chandra_SS433/3843/analysis/wavdetect_check/` (`2-7_*` from the
  wrapper, `cli/` for the comparison run).
