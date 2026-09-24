from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from xray_tools import (
    CIAOGTI,
    CIAOBlankSky,
    CIAOEvents,
    CIAOLightCurve,
    CIAOWorkflowError,
    merge_events,
)


class FakeColumn:
    def __init__(self, values):
        self.values = values


class FakeCrate:
    def __init__(self, columns):
        self.columns = columns

    def get_colnames(self):
        return list(self.columns)

    def get_column(self, name):
        available = {column.casefold(): column for column in self.columns}
        try:
            return FakeColumn(self.columns[available[name.casefold()]])
        except KeyError as exc:
            raise ValueError(f"Column {name!r} does not exist") from exc


class FakeVectorCrate(FakeCrate):
    def get_colnames(self):
        return ["time", "sky(x, y)", "status"]


@pytest.fixture
def event_file(tmp_path: Path) -> Path:
    path = tmp_path / "evt2.fits"
    path.touch()
    return path


def test_filter_is_immutable_and_builds_dm_expression() -> None:
    events = CIAOEvents("evt2.fits")
    selected = events.filter("energy=500:7000").filter("ccd_id=7")

    assert events.filters == ()
    assert selected.filters == ("energy=500:7000", "ccd_id=7")
    assert selected.dm == "evt2.fits[energy=500:7000][ccd_id=7]"
    assert selected.filename == Path("evt2.fits")


@pytest.mark.parametrize("bounds", [(7000, 500), (500, 500), (np.nan, 500)])
def test_energy_rejects_invalid_bounds(bounds: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="emin_ev"):
        CIAOEvents("evt2.fits").energy(*bounds)


def test_ccd_id_builds_valid_filter() -> None:
    assert CIAOEvents("evt2.fits").ccd_id(5, 6, 7).dm == ("evt2.fits[ccd_id=5,6,7]")

    with pytest.raises(ValueError, match="at least one"):
        CIAOEvents("evt2.fits").ccd_id()


def test_region_filters_require_files(tmp_path: Path) -> None:
    region = tmp_path / "source.reg"
    region.touch()
    events = CIAOEvents("evt2.fits")

    assert events.region(region).dm == f"evt2.fits[sky=region({region})]"
    assert events.exclude_region(region).dm == (
        f"evt2.fits[exclude sky=region({region})]"
    )

    with pytest.raises(FileNotFoundError, match="region file"):
        events.region(tmp_path / "missing.reg")


def test_apply_gti_uses_canonical_at_filter(tmp_path: Path) -> None:
    gti_path = tmp_path / "clean.gti"
    gti_path.touch()
    events = CIAOEvents("evt2.fits")

    from_product = events.apply_gti(CIAOGTI(gti_path))
    from_path = events.apply_gti(gti_path)

    assert from_product.dm == f"evt2.fits[@{gti_path}]"
    assert from_path.dm == from_product.dm
    assert events.filters == ()


def test_energy_and_region_exclusion_use_composable_dm_syntax(
    tmp_path: Path,
) -> None:
    region = tmp_path / "source.reg"
    region.touch()
    selected = CIAOEvents("evt2.fits").energy(2500, 7000).exclude_region(region)

    assert selected.filters == (
        "energy=2500:7000",
        f"exclude sky=region({region})",
    )
    assert selected.dm == (
        f"evt2.fits[exclude (sky=region({region}))||(energy=:2500,7000:)]"
    )


def test_ccd_and_region_exclusion_use_composable_dm_syntax(tmp_path: Path) -> None:
    region = tmp_path / "source.reg"
    region.touch()
    selected = CIAOEvents("evt2.fits").ccd_id(3).exclude_region(region)

    assert selected.dm == (
        f"evt2.fits[exclude (sky=region({region}))||(ccd_id=0,1,2,4,5,6,7,8,9)]"
    )


def test_unsupported_filter_and_region_exclusion_require_materialization(
    tmp_path: Path,
) -> None:
    region = tmp_path / "source.reg"
    region.touch()
    selected = CIAOEvents("evt2.fits").filter("grade=0").exclude_region(region)

    with pytest.raises(CIAOWorkflowError, match="Materialize"):
        _ = selected.dm


def test_write_materializes_selection_with_dmcopy(event_file: Path) -> None:
    output = event_file.parent / "clean_evt2.fits"
    events = CIAOEvents(event_file).energy(500, 7000)
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        result = events.write(output, clobber=True)

    runtool.dmcopy.assert_called_once_with(
        infile=f"{event_file}[energy=500:7000]",
        outfile=str(output),
        clobber=True,
    )
    assert result == CIAOEvents(output)


def test_write_refuses_implicit_overwrite(event_file: Path) -> None:
    output = event_file.parent / "clean_evt2.fits"
    output.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        CIAOEvents(event_file).write(output)


def test_lightcurve_calls_dmextract(event_file: Path) -> None:
    output = event_file.parent / "background.lc"
    events = CIAOEvents(event_file).energy(2500, 7000)
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        result = events.lightcurve(output, binsize=259.28, clobber=True)

    runtool.dmextract.assert_called_once_with(
        infile=f"{event_file}[energy=2500:7000][bin time=::259.28]",
        outfile=str(output),
        opt="ltc1",
        clobber=True,
    )
    assert result == CIAOLightCurve(output)


def test_lightcurve_rejects_non_positive_binsize() -> None:
    with pytest.raises(ValueError, match="binsize"):
        CIAOEvents("evt2.fits").lightcurve("background.lc", binsize=0)


def test_lightcurve_creates_missing_output_directories(event_file: Path) -> None:
    output = event_file.parent / "sub" / "dir" / "background.lc"
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        result = CIAOEvents(event_file).lightcurve(output, clobber=True)

    assert result == CIAOLightCurve(output)
    assert output.parent.is_dir()
    runtool.dmextract.assert_called_once_with(
        infile=f"{event_file}[bin time=::259.28]",
        outfile=str(output),
        opt="ltc1",
        clobber=True,
    )


def test_fluximage_serializes_energy_intervals(event_file: Path) -> None:
    runtool = Mock()
    outroot = event_file.parent / "ss433"
    discovered = {"0.5-2": object(), "2-7": object()}

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.events._discover_flux_images", return_value=discovered),
    ):
        result = CIAOEvents(event_file).fluximage(
            outroot,
            energy=[(0.5, 2.0), (2.0, 7.0)],
            binsize=4,
            clobber=True,
        )

    kwargs = runtool.fluximage.call_args.kwargs
    assert kwargs["infile"] == str(event_file)
    assert kwargs["outroot"] == str(outroot)
    assert kwargs["bands"] == "0.5:2:1,2:7:3.74166"
    assert kwargs["binsize"] == 4
    assert kwargs["clobber"] is True
    assert kwargs["units"] == "default"
    assert result is discovered


def test_fluximage_passes_bands_through_verbatim(event_file: Path) -> None:
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.events._discover_flux_images", return_value={"broad": object()}),
    ):
        CIAOEvents(event_file).fluximage("ss433", bands="csc,broad")

    kwargs = runtool.fluximage.call_args.kwargs
    assert kwargs["bands"] == "csc,broad"
    assert "binsize" not in kwargs


def test_fluximage_uses_the_lazy_dm_expression(event_file: Path) -> None:
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.events._discover_flux_images", return_value={"broad": object()}),
    ):
        CIAOEvents(event_file).energy(2500, 7000).fluximage("ss433")

    assert runtool.fluximage.call_args.kwargs["infile"] == (
        f"{event_file}[energy=2500:7000]"
    )


def test_fluximage_rejects_energy_and_bands_together(event_file: Path) -> None:
    with pytest.raises(ValueError, match="either energy or bands"):
        CIAOEvents(event_file).fluximage("ss433", energy=(0.5, 7.0), bands="broad")


@pytest.mark.parametrize(
    "energy",
    [(2.0, 0.5), (0.0, 7.0), (-1.0, 7.0), [(0.5, 2.0), (7.0, 2.0)]],
)
def test_fluximage_rejects_invalid_energy_intervals(
    event_file: Path,
    energy,
) -> None:
    runtool = Mock()
    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        pytest.raises(ValueError, match="0 < lo < hi"),
    ):
        CIAOEvents(event_file).fluximage("ss433", energy=energy)
    runtool.fluximage.assert_not_called()


def test_fluximage_creates_missing_outroot_directories(
    event_file: Path,
    tmp_path: Path,
) -> None:
    runtool = Mock()
    prefix = tmp_path / "new" / "ss433"
    directory = tmp_path / "outdir"

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.events._discover_flux_images", return_value={"broad": object()}),
    ):
        CIAOEvents(event_file).fluximage(prefix)
        CIAOEvents(event_file).fluximage(f"{directory}/")

    assert prefix.parent.is_dir()
    assert directory.is_dir()


def test_fluximage_raises_when_nothing_was_created(event_file: Path) -> None:
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.events._discover_flux_images", return_value={}),
        pytest.raises(CIAOWorkflowError, match="no flux images"),
    ):
        CIAOEvents(event_file).fluximage("ss433")


def test_to_crate_reads_the_lazy_expression(event_file: Path) -> None:
    crate = object()
    pycrates = Mock()
    pycrates.read_file.return_value = crate
    events = CIAOEvents(event_file).energy(500, 7000)

    with patch("xray_tools.events.require_pycrates", return_value=pycrates):
        result = events.to_crate()

    assert result is crate
    pycrates.read_file.assert_called_once_with(f"{event_file}[energy=500:7000]")


def test_to_pandas_selects_and_copies_columns(event_file: Path) -> None:
    time = np.array([1.0, 2.0])
    energy = np.array([500, 700])
    crate = FakeCrate({"TIME": time, "ENERGY": energy})
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.events.require_pycrates", return_value=pycrates):
        frame = CIAOEvents(event_file).to_pandas(columns=("time", "energy"))

    pycrates.read_file.assert_called_once_with(f"{event_file}[cols time,energy]")
    assert list(frame.columns) == ["time", "energy"]
    frame.loc[0, "time"] = 99
    assert time[0] == 1.0


def test_to_pandas_reads_virtual_vector_components(event_file: Path) -> None:
    x = np.array([10.0, 20.0])
    y = np.array([30.0, 40.0])
    crate = FakeVectorCrate({"x": x, "y": y})
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.events.require_pycrates", return_value=pycrates):
        frame = CIAOEvents(event_file).to_pandas(columns=("x", "y"))

    assert list(frame.columns) == ["x", "y"]
    assert frame.to_numpy().tolist() == [[10.0, 30.0], [20.0, 40.0]]


def test_to_pandas_expands_vectors_when_columns_are_not_given(
    event_file: Path,
) -> None:
    status = np.array([[False, True], [True, False]])
    crate = FakeVectorCrate(
        {
            "time": np.array([1.0, 2.0]),
            "x": np.array([10.0, 20.0]),
            "y": np.array([30.0, 40.0]),
            "status": status,
        }
    )
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.events.require_pycrates", return_value=pycrates):
        frame = CIAOEvents(event_file).to_pandas()

    assert list(frame.columns) == ["time", "x", "y", "status"]
    assert frame.loc[0, "status"].tolist() == [False, True]
    frame.loc[0, "status"][0] = True
    assert not status[0, 0]


def test_to_pandas_rejects_empty_column_selection(event_file: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        CIAOEvents(event_file).to_pandas(columns=())


def test_merge_events_stacks_inputs_in_order(event_file: Path, tmp_path: Path) -> None:
    other = tmp_path / "other_evt2.fits"
    other.touch()
    output = tmp_path / "merged" / "merged_evt2.fits"
    runtool = Mock()
    selected = CIAOEvents(event_file).energy(500, 7000)

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        result = merge_events([selected, other], output, clobber=True)

    runtool.dmmerge.assert_called_once_with(
        infile=f"{event_file}[energy=500:7000],{other}",
        outfile=str(output),
        clobber=True,
    )
    assert result == CIAOEvents(output)
    assert output.parent.is_dir()


def test_merge_events_requires_at_least_one_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        merge_events([], tmp_path / "merged.fits")


def test_merge_events_validates_input_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="event file"):
        merge_events([tmp_path / "missing.fits"], tmp_path / "merged.fits")


def test_merge_events_refuses_implicit_overwrite(event_file: Path) -> None:
    output = event_file.parent / "merged.fits"
    output.touch()
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        pytest.raises(FileExistsError, match="already exists"),
    ):
        merge_events([event_file], output)

    runtool.dmmerge.assert_not_called()


def test_blanksky_calls_the_blanksky_tool(event_file: Path, tmp_path: Path) -> None:
    output = event_file.parent / "blank" / "bkg_evt.fits"
    asol1 = tmp_path / "asol1.fits"
    asol2 = tmp_path / "asol2.fits"
    asol1.touch()
    asol2.touch()
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        result = CIAOEvents(event_file).blanksky(
            output,
            asolfile=[asol1, asol2],
            weight_method="particle-rate",
            bkgparams="[energy=8000:11000]",
            random=42,
            tmpdir=tmp_path,
            verbose=2,
            clobber=True,
        )

    runtool.blanksky.assert_called_once_with(
        evtfile=str(event_file),
        outfile=str(output),
        stowedbg=False,
        weight_method="particle-rate",
        bkgparams="[energy=8000:11000]",
        random=42,
        verbose=2,
        clobber=True,
        asolfile=f"{asol1},{asol2}",
        tmpdir=str(tmp_path),
    )
    assert result == CIAOBlankSky(output)
    assert output.parent.is_dir()


def test_blanksky_uses_the_lazy_dm_expression(event_file: Path) -> None:
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        CIAOEvents(event_file).energy(2500, 7000).blanksky("bkg_evt.fits", clobber=True)

    kwargs = runtool.blanksky.call_args.kwargs
    assert kwargs["evtfile"] == f"{event_file}[energy=2500:7000]"
    assert kwargs["stowedbg"] is False
    assert kwargs["weight_method"] == "particle"
    assert kwargs["bkgparams"] == "default"
    assert "asolfile" not in kwargs
    assert "tmpdir" not in kwargs


def test_blanksky_accepts_a_single_aspect_solution(event_file: Path, tmp_path: Path) -> None:
    asol = tmp_path / "asol1.fits"
    asol.touch()
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        CIAOEvents(event_file).blanksky("bkg_evt.fits", asolfile=asol)

    assert runtool.blanksky.call_args.kwargs["asolfile"] == str(asol)


@pytest.mark.parametrize("weight_method", ["particles", "rate", "PARTICLE"])
def test_blanksky_rejects_unknown_weight_methods(
    event_file: Path,
    weight_method: str,
) -> None:
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        pytest.raises(ValueError, match="weight_method"),
    ):
        CIAOEvents(event_file).blanksky("bkg_evt.fits", weight_method=weight_method)

    runtool.blanksky.assert_not_called()


@pytest.mark.parametrize(
    "bkgparams",
    ["[energy=abc]", "[time=1:2]", "9000:12000", "[pi=300:500:700]", "[energy=-1:5]"],
)
def test_blanksky_rejects_malformed_bkgparams(event_file: Path, bkgparams: str) -> None:
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        pytest.raises(ValueError, match="bkgparams"),
    ):
        CIAOEvents(event_file).blanksky("bkg_evt.fits", bkgparams=bkgparams)

    runtool.blanksky.assert_not_called()


@pytest.mark.parametrize("bkgparams", ["default", "[energy=9000:12000]", "[pi=300:500]"])
def test_blanksky_accepts_valid_bkgparams(event_file: Path, bkgparams: str) -> None:
    runtool = Mock()

    with patch("xray_tools.events.require_ciao_runtool", return_value=runtool):
        CIAOEvents(event_file).blanksky("bkg_evt.fits", bkgparams=bkgparams)

    assert runtool.blanksky.call_args.kwargs["bkgparams"] == bkgparams


def test_blanksky_refuses_implicit_overwrite(event_file: Path) -> None:
    output = event_file.parent / "bkg_evt.fits"
    output.touch()
    runtool = Mock()

    with (
        patch("xray_tools.events.require_ciao_runtool", return_value=runtool),
        pytest.raises(FileExistsError, match="already exists"),
    ):
        CIAOEvents(event_file).blanksky(output)

    runtool.blanksky.assert_not_called()


def test_blanksky_requires_an_event_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="event file"):
        CIAOEvents(tmp_path / "missing.fits").blanksky(tmp_path / "bkg_evt.fits")
