from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from xray_tools import CIAOGTI, CIAOEvents, CIAOLightCurve, CIAOWorkflowError


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
