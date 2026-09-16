from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from xray_tools import CIAOGTI, CIAOLightCurve


class FakeColumn:
    def __init__(self, values):
        self.values = values


class FakeCrate:
    def __init__(self, columns):
        self.columns = columns

    def get_colnames(self):
        return list(self.columns)

    def get_column(self, name):
        return FakeColumn(self.columns[name])


@pytest.fixture
def lightcurve_file(tmp_path: Path) -> Path:
    path = tmp_path / "background.lc"
    path.touch()
    return path


def test_product_paths_are_normalized() -> None:
    assert CIAOLightCurve("background.lc").path == Path("background.lc")
    assert CIAOGTI("background.gti").path == Path("background.gti")


def test_lightcurve_to_pandas_handles_optional_columns(
    lightcurve_file: Path,
) -> None:
    time = np.array([1.0, 2.0])
    rate = np.array([0.1, 0.2])
    crate = FakeCrate({"TIME": time, "RATE": rate})
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.products.require_pycrates", return_value=pycrates):
        frame = CIAOLightCurve(lightcurve_file).to_pandas()

    assert list(frame.columns) == ["time", "rate"]
    frame.loc[0, "rate"] = 10
    assert rate[0] == 0.1
    pycrates.read_file.assert_called_once_with(str(lightcurve_file))


def test_plot_uses_count_rate_and_returns_axes(lightcurve_file: Path) -> None:
    lightcurve = CIAOLightCurve(lightcurve_file)
    axes = Mock()
    frame = pd.DataFrame({"time": [1.0, 2.0], "count_rate": [0.1, 0.2]})

    with patch.object(CIAOLightCurve, "to_pandas", return_value=frame):
        result = lightcurve.plot(ax=axes)

    assert result is axes
    axes.plot.assert_called_once()
    axes.set_xlabel.assert_called_once_with("TIME")
    axes.set_ylabel.assert_called_once_with("Count rate")


def test_plot_requires_a_rate_column(lightcurve_file: Path) -> None:
    frame = pd.DataFrame({"time": [1.0, 2.0]})

    with (
        patch.object(CIAOLightCurve, "to_pandas", return_value=frame),
        pytest.raises(ValueError, match="COUNT_RATE or RATE"),
    ):
        CIAOLightCurve(lightcurve_file).plot(ax=Mock())


def test_clean_forwards_ciao_parameters(lightcurve_file: Path) -> None:
    output = lightcurve_file.parent / "background_clean.gti"
    lc_clean = Mock()

    with patch("xray_tools.products.require_lc_clean", return_value=lc_clean):
        result = CIAOLightCurve(lightcurve_file).clean(
            output,
            mean=0.2,
            clip=2.5,
            sigma=3.0,
            scale=1.5,
            minfrac=0.2,
            plot=False,
            verbose=2,
        )

    lc_clean.assert_called_once_with(
        str(lightcurve_file),
        outfile=str(output),
        mean=0.2,
        clip=2.5,
        sigma=3.0,
        scale=1.5,
        minfrac=0.2,
        plot=False,
        verbose=2,
    )
    assert result == CIAOGTI(output)


def test_clean_requires_a_real_lightcurve(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="light-curve file"):
        CIAOLightCurve(tmp_path / "missing.lc").clean(tmp_path / "clean.gti")
