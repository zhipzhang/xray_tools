# Design: image arithmetic via `dmimgcalc`, bound to `CIAOImage`

Status: implemented (2026-09-25); §6 records implementation notes and the
verification against CLI dmimgcalc.
References: [dmimgcalc ahelp](https://cxc.cfa.harvard.edu/ciao/ahelp/dmimgcalc.html).
Precondition verified (2026-09-19): `runtool.dmimgcalc` is registered in
`ciao_contrib.runtool` (CIAO 4.18).

Revision history: the first draft (2026-09-20) proposed explicit-`outfile`
methods `combine()`/`ratio()` and deferred dunders. The user decision of
2026-09-25 reverses that: the primary surface is the arithmetic operators
writing to temp files, with `save()` persisting results (§2, §3). The
`combine()`/`ratio()` methods are dropped; weighted/expression mode is
deferred (§5).

## 1. What `dmimgcalc` does, and what we must respect

`dmimgcalc infile infile2 outfile operation [weight] [weight2] [lookupTab] [clobber] [verbose]`

Two modes matter for us:

- **Simple two-image ops** — `operation` ∈ `{"add", "sub", "mul", "div"}`:
  `out = img1 <op> img2`. Output dtype is promoted to double when the input
  dtypes differ.
- **Arbitrary expressions** — `infile2="none"`,
  `operation="imgout=<dmmath expression>"` with `img1..imgN` references. For
  the operators this powers scalar operands: `img * 2` becomes
  `imgout=img1*2` in single-input mode.

Semantics we get for free by running the real tool (and would lose with numpy):

- **Invalid-data handling** exactly as CIAO defines it (division-by-zero →
  0/NaN per operation, NaN-input ops → 0, summary messages).
- **Header/WCS propagation** — output WCS comes from `infile`; keywords merge
  via the lookup-table rules (caveat: differing `DETNAM` merges to `"Merged"`).
- **Logical-coordinate combination** — the tool matches pixels by array
  position and *ignores WCS*, warning on mismatch. Shapes must be identical.
  → Alignment is the caller's responsibility; the wrapper pre-checks shape
  equality cheaply via `CIAOImage.shape` and documents that images must share
  a grid (e.g. come from the same fluximage run).

## 2. Proposed API

Public surface binds to `CIAOImage` (user decision, 2026-09-20; operator form
is the user decision of 2026-09-25): the components are the images
themselves, so the operators belong where the pixels are referenced. Works
for any aligned pair — flux bands, blanksky background images, exposure maps,
hand-made images. `CIAOFluxImage` is untouched (its components are already
`CIAOImage`s, e.g. `imgs["2-7"].flux`).

```python
# products/image.py — internal engine (module-level private helper)
def _dmimgcalc(
    infile1: str | Path,
    infile2: str | Path | None,      # None -> "none" (single-input/expression mode)
    outfile: str | Path,
    operation: str,                  # "add" | "sub" | "mul" | "div" | "imgout=..."
    *,
    clobber: bool = False,
    verbose: int = 0,
) -> Path:
    """Run CIAO dmimgcalc and return the output path."""

# CIAOImage — operators; each runs dmimgcalc into a temp file
def __add__(self, other: CIAOImage | str | Path | float) -> CIAOImage: ...
def __sub__(self, other): ...        # same operand types
def __mul__(self, other): ...
def __truediv__(self, other): ...
def __radd__/__rsub__/__rmul__/__rtruediv__(self, other): ...  # scalar left operands

# CIAOImage — persist a result
def save(self, outfile: str | Path, *, clobber: bool = False) -> CIAOImage:
    """Move a temp image produced by the operators to ``outfile``
    (a plain copy for any other image); return the saved CIAOImage."""
```

Typical workflow:

```python
imgs = evt.fluximage("out/i", energy=[(0.5, 2.0), (2.0, 7.0)], clobber=True)
bgs = blank.images("out/bkg", imgs["0.5-2"].counts)

net = imgs["0.5-2"].counts - bgs.background        # temp file, dmimgcalc "sub"
hardness = (imgs["2-7"].counts - imgs["0.5-2"].counts) / (
    imgs["2-7"].counts + imgs["0.5-2"].counts
)                                                   # chained temps
hardness.save("out/hardness.img", clobber=True)     # moved into place
hardness.plot(stretch="linear", vmin=-1, vmax=1)
doubled = (imgs["0.5-2"].exposure * 2).save("out/exp2.img")  # scalar -> imgout=img1*2
```

## 3. Design decisions

- **Run the tool, never numpy**: byte-identical behavior with CIAO
  command-line usage — NaN rules, dtype promotion, header merging. The package
  wraps, it does not reimplement.
- **Operators write temp files; `save()` makes them permanent** (user
  decision, 2026-09-25). This reverses the first draft's "no dunders" ruling:
  the old objection was that dunders force *implicit* outputs, clashing with
  the package's explicit-`outfile` convention. The temp-file + `save()`
  pattern resolves it — nothing lands in the user's directories until they
  name it, so the spirit of the convention holds.
- **Temp lifecycle**: `tempfile.mkstemp(prefix="xray_tools_", suffix=".img")`
  creates the file (dmimgcalc is then called with `clobber=True`, since
  mkstemp leaves an empty file). Each temp `CIAOImage` registers a
  `weakref.finalize` that unlinks its file when the image is garbage
  collected, so dropped intermediates in `a - b / c` chains clean themselves
  up. A module-level registry (`_TEMP_IMAGES`) plus an `atexit` hook removes
  any survivors at interpreter exit. `save()` detaches the finalizer before
  moving, so saved files are never cleaned.
- **`save()` moves temps, copies everything else** — the user's "think mv".
  Moving a package-owned temp is safe and avoids duplicating data;
  `shutil.move` handles the cross-filesystem case (TMPDIR vs data disk).
  Moving a *user* file would be destructive, so non-temp images are copied
  with `shutil.copy`. Both return `CIAOImage(outfile)`; `clobber` follows the
  usual `prepare_output_path` rules (temp file left in place on
  `FileExistsError`).
- **Operand types**: `CIAOImage`, `str | Path` (wrapped in `CIAOImage`), or a
  number (`numbers.Number`, so numpy scalars work). Numbers take dmimgcalc's
  expression mode: `img1 <op> <value>` (or reflected). Reflected image
  operands are not supported (return `NotImplemented`); reflected scalars are.
- **Shape pre-check before running** (image operands only):
  `self.shape == other.shape`, else `ValueError`. Catches the classic
  "bands from different runs/grids" mistake without a CIAO round trip. WCS
  equality is *not* pre-checked (pycrates transform introspection is
  unreliable; see image-design.md §2) — documented instead.
- **No `weight`/`weight2`, no public expression mode**: operators can't carry
  them, and the old `combine()`/`ratio()` methods that could are dropped to
  keep one obvious way to do arithmetic. The engine keeps the `None →
  "none"` single-input mapping, so a future named method for weighted or
  free-form expressions stays a thin addition (§5).
- `lookupTab` is not exposed (default header-merge rules are right for
  same-observation products); add if cross-observation merging ever lands.

## 4. Implementation steps

1. **products/image.py**
   - `_dmimgcalc(...)`: `require_file` both inputs (when not None),
     `prepare_output_path(outfile, clobber=clobber)`, `require_ciao_runtool()`,
     map `infile2=None → "none"`, run `runtool.dmimgcalc`, return the `Path`.
   - Temp plumbing: `_TEMP_IMAGES` registry, `_unlink_temp` finalizer
     callback, `_cleanup_temp_images` + `atexit.register`.
   - `CIAOImage._combine(other, operation, *, reflected=False)`: the shared
     operator body — number → `imgout=` expression, image-like → shape
     pre-check + simple op, else `NotImplemented`; mkstemp the output,
     delegate to `_dmimgcalc(clobber=True)`, register the finalizer, return
     the temp `CIAOImage`. The eight dunders are one-liners over it.
   - `CIAOImage.save(...)`: as in §2.
2. **Tests** (mock `xray_tools.products.image.require_ciao_runtool` +
   `require_pycrates`, no CIAO):
   - each operator asserts exact dmimgcalc kwargs (`operation`, temp
     `outfile`, `infile2="none"` for scalars, `clobber=True`);
   - scalar and reflected-scalar expressions (`img * 2` → `imgout=img1*2`,
     `10 - img` → `imgout=10-img1`);
   - shape mismatch → `ValueError`; unsupported operand type → `TypeError`;
   - result path registered in `_TEMP_IMAGES`; GC/`del` unlinks the temp;
     `_cleanup_temp_images` removes leftovers;
   - `save()` moves a temp (source gone, finalizer detached, returns saved
     `CIAOImage`), copies a non-temp (source stays), raises `FileExistsError`
     without clobber and leaves the temp intact.
3. **README**: operator example after the blanksky block (net image from
   `counts - bgs.background`, then `save()`).
4. **Verify on real data** (SS433 bands in
   `.../Chandra_SS433/3843/analysis/flux/`): the chained hardness map agrees
   with the equivalent CLI `dmimgcalc` runs, and `save()` moves the temp.

## 5. Open questions

- **Weighted ops / free-form expressions**: a named method (e.g.
  `combine(other, "imgout=...", outfile)`) remains possible over the same
  engine; add only when a workflow needs weights or `sqrt(...)`-style
  expressions that operators can't express.
- **Temp location**: system temp dir (TMPDIR) is the default; an image-sized
  file per operation may be better next to the inputs on some systems.
  Revisit if /tmp pressure shows up.
- **`CIAOFluxImage`-level operators** (band-minus-band in one call): sugar
  over component operators; defer until asked for.
- **Cross-observation arithmetic**: needs reprojection first
  (`reproject_image_grid`); out of scope until a workflow demands it.

## 6. Implementation notes (2026-09-25)

- All steps in §4 landed; 10 new tests in `tests/test_products.py`, full suite
  green (138 passed). No deviations from the revised design.
- **Verified end-to-end on real data** (SS 433, ObsID 3843,
  `.../analysis/flux/` counts images, 405×485): the chained
  `(hard - soft) / (hard + soft)` hardness map and the scalar `hard * 2` are
  **pixel-identical** (`allclose(rtol=0, atol=0, equal_nan=True)`) to the
  equivalent CLI `dmimgcalc` runs (`sub`/`add`/`div` chain and
  `imgout=img1*2`). `save()` moved the `/tmp/xray_tools_*.img` temp into
  place; the intermediate `(hard ± soft)` temps were cleaned by their
  finalizers. Products left in `.../analysis/dmimgcalc_check/`
  (`hardness.img`, `hard_x2.img`, plus `*_cli.img` for the comparison).
- The temp file is created by `tempfile.mkstemp` before dmimgcalc runs, so
  the engine is always called with `clobber=True` from the operators;
  `save()` remains the only clobber-guarded write.
