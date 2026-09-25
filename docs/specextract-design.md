# Design: `specextract` support in xray_tools

Status: implemented (2026-09-25); §8 records implementation notes and the
scope decisions made at implementation time (ACIS, single-observation only —
`dtffile`/HRC and stacks/`combine` dropped from the core surface).
Reference: [specextract ahelp](https://cxc.harvard.edu/ciao/ahelp/specextract.html)
(CIAO 4.18), [sherpa PHA docs](https://sherpa.readthedocs.io/en/4.16.1/data/pha.html).

## 1. What specextract does

`specextract` is a CIAO **script** that chains the full spectral-extraction
pipeline for imaging-mode (and zeroth-order grating) data:

- `dmextract` — source (and optional background) PI spectrum, with an embedded
  WMAP (weight map) in the PHA primary header;
- `mkwarf` (weighted, `weight=yes`) or `mkarf` (unweighted, `weight=no`) — ARF;
  `sky2tdet` prepares the TDET WMAP for the weighted branch;
- `mkacisrmf` or `mkrmf` (chosen automatically via the CALDB `P2_RESP` lookup) — RMF;
- `arfcorr` — optional PSF aperture correction on unweighted ARFs
  (`correctpsf=yes`, output `*.corr.arf`);
- `dmgroup` — optional grouped copies (`*_grp.pi`);
- `combine_spectra` — optional combination of stacked inputs (`combine=yes`);
- header bookkeeping (`dmhedit`): `BACKFILE` / `ANCRFILE` / `RESPFILE` in the
  source (and background) spectra are set so downstream software (Sherpa)
  auto-loads everything; Galactic `Bell_nH` / `NRAO_nH` keywords are added.

The four user decisions the tool is built around (from the ahelp):

1. weighted vs unweighted responses (`weight`, `weight_rmf`) — extended vs
   point source;
2. PSF aperture correction when `weight=no` (`correctpsf`, `refcoord`);
3. background spectrum + optional background responses (`bkgfile`, `bkgresp`),
   linked into the source spectrum header;
4. combining stacked inputs (`combine`).

Ancillary files (`asp`, `mskfile`, `badpixfile`, `dtffile`, `dafile`) are
**auto-located from the event-file header when left blank** — the common case
needs none of them.

## 2. Parameter triage: what the wrapper exposes

specextract has 27 parameters. The package premise (user decision, 2026-09-25)
is to expose only the core set as explicit keyword arguments and leave the rest
at CIAO defaults. `infile` is `self` (the lazy `CIAOEvents` view — its `.dm`
expression already carries the `sky=region(...)` extraction filter, see §4).

### Core (explicit kwargs)

| specextract param | proposed kwarg | default | why core |
| --- | --- | --- | --- |
| `infile` | `self.dm` | — | the event view itself |
| `outroot` | `outroot` (positional) | — | required; same convention as `fluximage()`/`blanksky()` |
| `bkgfile` | `bkgfile` | `None` | background spectrum; accepts `CIAOEvents \| CIAOBlankSky \| str \| Path` (§4) |
| `weight` | `weight` | `True` | the extended-vs-point-source switch |
| `weight_rmf` | `weight_rmf` | `False` | weighted RMF is expensive, few-% gain |
| `correctpsf` | `correctpsf` | `False` | aperture correction for point sources |
| `refcoord` | `refcoord` | `None` | response position; accept `"RA Dec"` string or `(ra, dec)` tuple |
| `grouptype` | `grouptype` | `"NUM_CTS"` | grouping is routine analysis input |
| `binspec` | `binspec` | `15` | paired with `grouptype` |
| `bkg_grouptype` | `bkg_grouptype` | `"NONE"` | background grouping |
| `bkg_binspec` | `bkg_binspec` | `None` | paired with `bkg_grouptype` |
| `bkgresp` | `bkgresp` | `True` | background responses; auto-forced `False` for blank-sky input (§4) |
| `asp` | `asolfile` | `None` = header lookup | ancillary override (rare, but the package already uses the `asolfile` name in `fluximage()`/`blanksky()`); CIAO stack syntax works through the string form |
| `mskfile` | `maskfile` | `None` = header lookup | ancillary override; name matches `fluximage()` |
| `badpixfile` | `badpixfile` | `None` = header lookup | ancillary override |
| `dafile` | `dafile` | `None` = CALDB | dead-area correction toggle (`"NONE"` to disable) |
| `energy` | `energy` | `"0.3:11.0:0.01"` | response grid; rarely changed, cheap to expose |
| `channel` | `channel` | `"1:1024:1"` | PI channel grid; non-default values also filter the spectrum |
| `parallel` | `parallel` | `True` | matches `fluximage()` |
| `nproc` | `nproc` | `None` | matches `fluximage()` |
| `tmpdir` | `tmpdir` | `None` | matches `blanksky()` |
| `verbose` | `verbose` | `0` | housekeeping |
| `clobber` | `clobber` | `False` | housekeeping |

### Excluded (stay at CIAO defaults; add later only on demand)

| specextract param | default | why excluded |
| --- | --- | --- |
| `rmffile` | `CALDB` | CALDB directive for RMF-tool selection; overriding it is expert territory |
| `resp_pos` | `REGION` | position-refinement method for unweighted responses; default is the sensible modern behavior |
| `combine` | `no` | only meaningful for stacked multi-obs input — deferred with stack support (§5) |
| `dtffile` | header lookup | HRC-only (dead-time factor); out of the ACIS scope decided 2026-09-25 |
| `readout_streakspec` | `no` | niche: ACIS readout-streak spectra |
| `energy_wmap` | `300:2000` | WMAP energy band (eV); default fits typical source emission |
| `binwmap` | `tdet=8` | RMF WMAP binning |
| `binarfwmap` | `1` | ARF WMAP binning; only needed when mkwarf hits memory limits on chip-scale regions |
| `binarfcorr` | `1` | arfcorr pixel binning; only for sub-pixel work |

No `**kwargs` passthrough: the package style is explicit options with early
validation (`fluximage()`, `blanksky()`), and a passthrough would defeat that.
The exclusion list is one line each to revisit.

## 3. Proposed API

```python
# events.py — on CIAOEvents (same shape as lightcurve() / fluximage() / blanksky())
evt.specextract(
    outroot,
    *,
    bkgfile=None,          # CIAOEvents | CIAOBlankSky | str | Path (region file)
    weight=True,
    weight_rmf=False,
    correctpsf=False,
    refcoord=None,         # "RA Dec" | (ra, dec) in decimal degrees
    grouptype="NUM_CTS",
    binspec=15,
    bkg_grouptype="NONE",
    bkg_binspec=None,
    bkgresp=True,
    asolfile=None, maskfile=None, badpixfile=None, dafile=None,
    energy="0.3:11.0:0.01",
    channel="1:1024:1",
    parallel=True, nproc=None, tmpdir=None,
    verbose=0, clobber=False,
) -> CIAOSpectrum
```

```python
# products/spectrum.py — new file-backed product
@dataclass(frozen=True)
class CIAOSpectrum:
    source: Path                    # {outroot}.pi (ungrouped)
    arf: Path | None                # .arf, or .corr.arf when correctpsf=True
    rmf: Path | None
    background: Path | None         # {outroot}_bkg.pi
    grouped_source: Path | None     # *_grp.pi when grouping requested
    grouped_background: Path | None
    bkg_arf: Path | None            # only when bkgresp=True
    bkg_rmf: Path | None

    # lazy metadata via require_pycrates() — same read pattern as CIAOBlankSky.scales
    exposure: float        # EXPOSURE keyword (cached_property)
    backscal: float        # BACKSCAL
    counts: int            # total source counts (TOTCTS / sum of COUNTS)

    def to_sherpa(self, *, prefer_grouped: bool = True, use_errors: bool = False):
        """Read into sherpa.astro.io.DataPHA via read_pha (see §4)."""
```

Typical workflow:

```python
evt = CIAOEvents("evt2.fits")
spec = (evt
        .region("src.reg")
        .specextract("extract/3c273", bkgfile="bkg.reg",
                     weight=False, correctpsf=True,
                     grouptype="NUM_CTS", binspec=15))
pha = spec.to_sherpa()          # DataPHA with ARF/RMF/background attached
```

## 4. Sherpa integration: return `DataPHA` directly, or a file-backed wrapper?

User idea (2026-09-25): the extraction is a natural point to connect with
Sherpa — return a `sherpa.astro.io.read_pha(...)` object (note: the module is
`sherpa.astro.io`, not `sherpa.astropy.io`) or a thin wrapper around it.

Verified fact that shapes this decision
([sherpa PHA docs](https://sherpa.readthedocs.io/en/4.16.1/data/pha.html)):
`sherpa.astro.io.read_pha()` **auto-loads the ARF, RMF, and background** listed
in the PHA header keywords, and specextract maintains exactly those keywords
(`ANCRFILE`, `RESPFILE`, `BACKFILE`). So any route back into Sherpa is a
one-liner once the files exist — the question is only what `specextract()`
itself returns.

### Option A: return `sherpa.astro.data.DataPHA` directly

- Pros: one call from events to a fit-ready object; shortest possible notebook
  flow; auto-attached responses make it genuinely complete.
- Cons: breaks the package's file-backed, lazy pattern — every existing product
  (`CIAOFluxImage`, `CIAOLightCurve`, `CIAOBlankSky`, `CIAOGTI`) holds paths and
  reads on demand; a `DataPHA` is eagerly in-memory. Requires importing sherpa
  at extraction time even when the user only wants the files (sherpa ships with
  CIAO, so this is availability-in-env, but it is still a heavier import than
  pycrates). Which file gets read (grouped vs ungrouped) becomes hidden policy.
  The other outputs specextract wrote (ungrouped/grouped variants, background
  responses, `.corr.arf`) become unreachable from the return value. Couples the
  public API to a third-party class's evolution; mocking in tests needs a fake
  `DataPHA` rather than a fake runtool.

### Option B: return a wrapper holding a `DataPHA`

- Pros: keeps a package-owned type; could add convenience properties.
- Cons: all of Option A's cons (still eager, still sherpa at call time), plus
  more code, plus a wrapper that either leaks the wrapped object everywhere or
  re-implements DataPHA's large surface. Worst of both worlds unless the wrapper
  adds real behavior — and the behaviors worth having (exposure, backscal,
  paths) are file metadata, which a path-backed object already provides.

### Option C (recommended): file-backed `CIAOSpectrum` + `to_sherpa()`

- Pros: consistent with the whole package; lazy; fully testable with the
  existing mock seams (`require_ciao_runtool`, `require_pycrates`); sherpa is
  touched only inside `to_sherpa()` behind a new `require_sherpa()` seam —
  mirroring `require_pycrates()` / `require_lc_clean()`; every output file
  stays addressable (grouped vs ungrouped is an explicit argument, not policy);
  metadata properties (`exposure`, `backscal`, `counts`) come from the PHA
  header via pycrates, no sherpa needed; `plot()` can be added later on the
  same footing as `CIAOLightCurve.plot`.
- Cons: two lines instead of one (`spec = evt...specextract(...)` then
  `pha = spec.to_sherpa()`); the returned `DataPHA` is a snapshot — re-running
  specextract with `clobber=True` requires calling `to_sherpa()` again (minor;
  the same is true of `to_pandas()` elsewhere).

`to_sherpa(prefer_grouped=True)` reads `grouped_source` when grouping was
requested, else `source`; passes through to
`sherpa.astro.io.read_pha(str(path), use_errors=use_errors)`. Background and
responses attach automatically from the header keywords — nothing to wire up.

## 5. Design notes and decisions

- **Naming: `bkgfile`, not `bkg_input`.** The package keeps CIAO parameter
  names where they exist (`asolfile`, `badpixfile`, `maskfile`, `dtffile` in
  `fluximage()`; `weight_method`, `bkgparams` in `blanksky()`), and `background`
  is already taken by `fluximage(background=...)` with a different meaning
  (method selector). Same for `weight` (not `weights`) — matches the ahelp, so
  ahelp examples translate mechanically.
- **Region ergonomics come free.** The source region is just the lazy filter:
  `evt.region("src.reg")` puts `sky=region(src.reg)` into `self.dm`, which is
  exactly the `infile` syntax specextract wants. Inline filters
  (`evt.filter("sky=circle(4123.5,3987.2,12)")`) work identically. The method
  should warn-or-error when no `sky=` filter is present, since a full-chip
  spectrum is almost never intended (mkwarf may also hit its memory limit there).
- **`bkgfile` accepts three forms.**
  - `CIAOEvents` — use its `.dm` verbatim (full control: different event file,
    extra filters, e.g. `CIAOEvents("bkg_evt.fits").region("bkg.reg")`);
  - `CIAOBlankSky` — use its `.path` as the background event file and **force
    `bkgresp=False`**, since specextract errors out when asked to build
    responses for blank-sky files (ahelp `bkgresp` / Dec 2013 change). This is
    the natural continuation of the existing `blanksky()` workflow;
  - `str | Path` — treated as a region file on the same underlying event file:
    `f"{self.filename}[sky=region({path})]"` (the most common case; validated
    with `require_file`). Note this composes from `self.filename`, not
    `self.dm`, so the source-region filter does not leak into the background.
- **Defaults kept.** `grouptype="NUM_CTS", binspec=15` are CIAO's defaults;
  faint-source users will often want `binspec=1` or `grouptype="NONE"`, but
  diverging from the tool defaults is a surprise of its own. Document, don't
  override.
- **Scope: single observation, ACIS only** (user decision, 2026-09-25).
  `CIAOEvents` wraps one event file, so stacks (`@lis` input, per-stack
  `outroot`, `combine=`) are out of scope for this method. A future
  `combine_spectra` wrapper (or a module-level function taking
  `Sequence[CIAOEvents]`) is the place for multi-obs work. HRC support
  (`dtffile`) is likewise dropped from the core surface.
- **Output discovery.** With a single input the outputs are deterministic and
  carry **no** `_src1`/`_bkg1` indices (those appear only for stacked inputs —
  verified against a real run, see §8): `{outroot}.pi`, `{outroot}.arf` (plus
  `{outroot}.corr.arf` when `correctpsf=yes`), `{outroot}.rmf`,
  `{outroot}_bkg.pi` / `_bkg.arf` / `_bkg.rmf`, and `{outroot}_grp.pi` /
  `{outroot}_bkg_grp.pi` when grouping. Discovery still filters by mtime ≥
  run-start, mirroring `_discover_flux_images`, so products from earlier runs
  with the same outroot are left out.

## 6. Implementation steps

1. Verify `runtool.specextract` is registered in `ciao_contrib.runtool`
   (CIAO 4.18) — same one-line check done for blanksky.
2. `utils.py`: add `require_sherpa()` (import `sherpa.astro.io`, raise
   `CIAOWorkflowError` with the "run inside CIAO" message otherwise).
3. `products/spectrum.py`: `CIAOSpectrum` dataclass + `_discover_spectrum()`
   glob helper + metadata cached_properties via `require_pycrates()` +
   `to_sherpa()` via `require_sherpa()`.
4. `events.py`: `specextract()` method — early validation (`binspec` consistent
   with `grouptype`, `refcoord` parsing, `bkgfile` form dispatch, sky-filter
   presence check), conditional-options dict, `runtool.specextract(...)`,
   function-local import of `CIAOSpectrum` (avoids the events↔products cycle,
   as in `lightcurve()`).
5. Exports in `products/__init__.py` and `xray_tools/__init__.py`.
6. Tests (existing mock patterns, no CIAO needed): `test_events.py` — patched
   `require_ciao_runtool`, assert exact kwargs and validation errors;
   `test_products.py` — patched `require_pycrates`/`require_sherpa`, discovery
   via tmp files + `os.utime`, `to_sherpa` grouped/ungrouped selection.
7. README: short example as in §3.

## 7. Open questions

- `CIAOSpectrum.plot()` — quick-look grouped spectrum with matplotlib (counts
  vs channel, or energy when RMF present)? Useful but not required for v1.
- Should `to_sherpa()` also expose `subtract()`-style background handling, or
  leave all of that to Sherpa (recommended — do not re-wrap Sherpa's API)?
- When stack support arrives: separate `combine_spectra` function, or
  `specextract()` accepting `Sequence[CIAOEvents]`? Defer with `combine`.

## 8. Implementation notes (2026-09-25)

- All steps in §6 landed and were verified end-to-end on real data (ObsID
  13858, `weight=False`, `correctpsf=True`, background with grouping,
  2026-09-25). The verification corrected two naming assumptions taken from
  the ahelp: single-input outputs carry **no** `_src1`/`_bkg1` indices
  (`{outroot}.pi`, `{outroot}_bkg.pi`, `{outroot}_bkg_grp.pi` — the numbered
  suffixes are stack-only), and a `correctpsf=yes` run writes **both** the
  plain `.arf` and the `.corr.arf` (discovery prefers the latter, matching
  the ANCRFILE keyword the PHA header points at). Also confirmed: `outroot`
  is purely a file prefix (`"spectrum/3843"` → `spectrum/3843.pi`, ...); the
  directory-style (trailing-`/`) branch carried over from `fluximage()` was
  removed and now raises `ValueError`. `to_sherpa()` on the run's products
  returned a `DataPHA` with the corrected ARF, RMF, and grouped background
  auto-attached from the header keywords.
- `dtffile` was dropped from the signature (HRC scope cut); the "ACIS-first,
  HRC not blocked" bullet from the proposal became the scope decision in §5.
- `CIAOSpectrum` fields after `source` all default to `None`, so discovery and
  tests can construct partial products; `__post_init__` normalizes every field
  to `Path`.
- No sky-filter presence **error** — the method logs a warning and proceeds
  (a full-chip extraction is unusual but valid).
- `_discover_spectrum` checks `.corr.arf` before `.arf`, so a PSF-corrected ARF
  wins when both are present; the mtime filter keeps stale products from
  earlier runs out.
- `to_sherpa()` returns `sherpa.astro.io.read_pha(str(path),
  use_errors=use_errors)` — responses and background attach from the PHA
  header keywords, nothing else is wired up.
