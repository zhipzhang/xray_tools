import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from xray_tools import CIAOFluxImage, CIAOGTI, CIAOLightCurve, LightCurveCuts
from xray_tools.products import _discover_flux_images, _parse_userlimit_cuts


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
    axes.set_xlabel.assert_called_once_with("Elapsed time (ks)")
    axes.set_ylabel.assert_called_once_with("Count rate (count/s)")


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


def test_clean_never_plots(lightcurve_file: Path) -> None:
    output = lightcurve_file.parent / "background_clean.gti"
    lc_clean = Mock()

    with (
        patch("xray_tools.products.require_lc_clean", return_value=lc_clean),
        patch.object(CIAOLightCurve, "plot") as plot,
    ):
        result = CIAOLightCurve(lightcurve_file).clean(output)

    assert lc_clean.call_args.kwargs["plot"] is False
    plot.assert_not_called()
    assert result == CIAOGTI(output)


def test_lightcurve_to_pandas_copies_rate_errors(lightcurve_file: Path) -> None:
    time = np.array([1.0, 2.0])
    rate = np.array([0.1, 0.2])
    error = np.array([0.01, 0.02])
    crate = FakeCrate({"TIME": time, "COUNT_RATE": rate, "COUNT_RATE_ERR": error})
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.products.require_pycrates", return_value=pycrates):
        frame = CIAOLightCurve(lightcurve_file).to_pandas()

    assert list(frame.columns) == ["time", "count_rate", "count_rate_err"]


def test_plot_draws_steps_and_errorbars(lightcurve_file: Path) -> None:
    frame = pd.DataFrame(
        {
            "time": [1.5, 2.5],
            "time_min": [1.0, 2.0],
            "time_max": [2.0, 3.0],
            "count_rate": [0.1, 0.2],
            "count_rate_err": [0.01, 0.02],
        }
    )
    axes = Mock()

    with patch.object(CIAOLightCurve, "to_pandas", return_value=frame):
        result = CIAOLightCurve(lightcurve_file).plot(ax=axes)

    assert result is axes
    (x, y), kwargs = axes.plot.call_args
    assert kwargs["drawstyle"] == "steps-post"
    assert list(x) == [0.0, 0.001, 0.002]
    assert list(y) == [0.1, 0.2, 0.2]
    (err_x, err_y, *_), err_kwargs = axes.errorbar.call_args
    assert list(err_x) == [0.0005, 0.0015]
    assert err_kwargs["yerr"] is frame["count_rate_err"]
    axes.set_xlabel.assert_called_once_with("Elapsed time (ks)")


def test_plot_overlays_gti_regions_and_cuts(
    lightcurve_file: Path,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [1500.0, 2500.0],
            "time_min": [1000.0, 2000.0],
            "time_max": [2000.0, 3000.0],
            "count_rate": [0.1, 0.2],
        }
    )
    intervals = pd.DataFrame(
        {"start": [1200.0, 2200.0], "stop": [1700.0, 2700.0], "duration": [500.0, 500.0]}
    )
    cuts = LightCurveCuts("count_rate", 0.05, 0.3)
    axes = Mock()
    gti = CIAOGTI(tmp_path / "clean.gti")

    with (
        patch.object(CIAOLightCurve, "to_pandas", return_value=frame),
        patch.object(CIAOGTI, "to_pandas", return_value=intervals),
        patch.object(CIAOGTI, "cuts", cuts),
    ):
        CIAOLightCurve(lightcurve_file).plot(ax=axes, gti=gti)

    assert axes.axvspan.call_count == 2
    assert [call.args for call in axes.axhline.call_args_list] == [(0.05,), (0.3,)]
    _, _, annotation = axes.text.call_args.args
    assert "Total observation: 2.00 ks" in annotation
    assert "GTI good time: 1.00 ks (50.0%)" in annotation


def test_plot_sets_a_given_title(lightcurve_file: Path) -> None:
    frame = pd.DataFrame({"time": [1.0, 2.0], "count_rate": [0.1, 0.2]})
    axes = Mock()

    with patch.object(CIAOLightCurve, "to_pandas", return_value=frame):
        CIAOLightCurve(lightcurve_file).plot(ax=axes, title="SS 433, ObsID 3843")

    axes.set_title.assert_called_once_with("SS 433, ObsID 3843")


def test_gti_to_pandas_copies_intervals(tmp_path: Path) -> None:
    start = np.array([1.0, 5.0])
    stop = np.array([3.0, 6.0])
    crate = FakeCrate({"START": start, "STOP": stop})
    pycrates = Mock()
    pycrates.read_file.return_value = crate
    gti_file = tmp_path / "clean.gti"
    gti_file.touch()

    with patch("xray_tools.products.require_pycrates", return_value=pycrates):
        frame = CIAOGTI(gti_file).to_pandas()

    assert list(frame.columns) == ["start", "stop", "duration"]
    assert frame["duration"].tolist() == [2.0, 1.0]
    pycrates.read_file.assert_called_once_with(f"{gti_file}[GTI]")


def test_gti_to_pandas_requires_interval_columns(tmp_path: Path) -> None:
    crate = FakeCrate({"START": np.array([1.0])})
    pycrates = Mock()
    pycrates.read_file.return_value = crate
    gti_file = tmp_path / "clean.gti"
    gti_file.touch()

    with (
        patch("xray_tools.products.require_pycrates", return_value=pycrates),
        pytest.raises(ValueError, match="START and STOP"),
    ):
        CIAOGTI(gti_file).to_pandas()


def test_gti_cuts_reads_filter_history(tmp_path: Path) -> None:
    gti_file = tmp_path / "clean.gti"
    gti_file.touch()
    crate = Mock()
    crate.get_history_records.return_value = [
        ["HISTORY", " PARM  :userlimit=(count_rate>0.1 && count_rate<0ASC00002", "HISTNUM"],
        ["HISTORY", " CONT  :.5)                                                      ASC00003", "HISTNUM"],
    ]
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.products.require_pycrates", return_value=pycrates):
        cuts = CIAOGTI(gti_file).cuts

    pycrates.read_file.assert_called_once_with(f"{gti_file}[FILTER]")
    assert cuts == LightCurveCuts("count_rate", 0.1, 0.5)


def test_gti_cuts_is_none_without_filter_history(tmp_path: Path) -> None:
    gti_file = tmp_path / "plain.gti"
    gti_file.touch()
    pycrates = Mock()
    pycrates.read_file.side_effect = OSError("no FILTER block")

    with patch("xray_tools.products.require_pycrates", return_value=pycrates):
        assert CIAOGTI(gti_file).cuts is None


def test_gti_plot_draws_interval_bars(tmp_path: Path) -> None:
    intervals = pd.DataFrame(
        {"start": [10.0, 20.0], "stop": [15.0, 22.0], "duration": [5.0, 2.0]}
    )
    axes = Mock()
    gti = CIAOGTI(tmp_path / "clean.gti")

    with patch.object(CIAOGTI, "to_pandas", return_value=intervals):
        result = gti.plot(ax=axes)

    assert result is axes
    (xranges, yrange), _ = axes.broken_barh.call_args
    assert list(xranges) == [(0.0, 0.005), (0.01, 0.002)]
    assert yrange == (0, 1)


def test_parse_userlimit_cuts_recovers_split_records() -> None:
    records = [
        ["HISTORY", " PARM  :userlimit=(count_rate>0.4165853273710954 && count_rate<0ASC00006", "HISTNUM"],
        ["HISTORY", " CONT  :.5998828714143773)                                      ASC00007", "HISTNUM"],
    ]

    cuts = _parse_userlimit_cuts(records)

    assert cuts == LightCurveCuts("count_rate", 0.4165853273710954, 0.5998828714143773)


def test_parse_userlimit_cuts_without_rate_limit() -> None:
    assert _parse_userlimit_cuts([["HISTORY", " PARM  :smooth=yes", "HISTNUM"]]) is None
    assert _parse_userlimit_cuts([]) is None


def _touch(path: Path, *, stale: bool = False) -> Path:
    path.touch()
    if stale:
        moment = time.time() - 3600
        os.utime(path, (moment, moment))
    return path


def test_discover_flux_images_maps_products_and_skips_stale(tmp_path: Path) -> None:
    flux = _touch(tmp_path / "ss433_broad_flux.img")
    counts = _touch(tmp_path / "ss433_broad_thresh.img")
    exposure = _touch(tmp_path / "ss433_broad_thresh.expmap")
    fov = _touch(tmp_path / "ss433_3843.fov")
    _touch(tmp_path / "ss433_soft_flux.img", stale=True)
    _touch(tmp_path / "ss433_soft_thresh.img", stale=True)
    _touch(tmp_path / "ss433_soft_thresh.expmap", stale=True)

    images = _discover_flux_images(tmp_path / "ss433", since=time.time() - 5)

    assert list(images) == ["broad"]
    assert images["broad"] == CIAOFluxImage(
        band="broad",
        counts=counts,
        flux=flux,
        exposure=exposure,
        fov=fov,
        psf=None,
    )


def test_discover_flux_images_falls_back_to_unthresholded_names(tmp_path: Path) -> None:
    _touch(tmp_path / "ss433_hard_flux.img")
    counts = _touch(tmp_path / "ss433_hard.img")
    exposure = _touch(tmp_path / "ss433_hard.expmap")

    images = _discover_flux_images(tmp_path / "ss433", since=time.time() - 5)

    assert images["hard"].counts == counts
    assert images["hard"].exposure == exposure
    assert images["hard"].fov is None
    assert images["hard"].psf is None


def test_discover_flux_images_finds_psf_maps(tmp_path: Path) -> None:
    _touch(tmp_path / "ss433_broad_flux.img")
    psf = _touch(tmp_path / "ss433_broad_thresh.psfmap")

    images = _discover_flux_images(tmp_path / "ss433", since=time.time() - 5)

    assert images["broad"].psf == psf


def test_discover_flux_images_with_directory_outroot(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    flux = _touch(outdir / "0.5-2_flux.img")
    fov = _touch(outdir / "3843.fov")

    images = _discover_flux_images(f"{outdir}/", since=time.time() - 5)

    assert list(images) == ["0.5-2"]
    product = images["0.5-2"]
    assert product.band == "0.5-2"
    assert product.flux == flux
    assert product.fov == fov


def _write_image(path: Path, data) -> Path:
    from astropy.io import fits

    fits.PrimaryHDU(np.asarray(data, dtype=float)).writeto(path)
    return path


def _flux_product(tmp_path: Path, **files: Path) -> CIAOFluxImage:
    defaults = {"counts": None, "flux": None, "exposure": None, "fov": None, "psf": None}
    defaults.update(files)
    return CIAOFluxImage(band="broad", **defaults)


def test_flux_image_plot_displays_flux_with_log_stretch(tmp_path: Path) -> None:
    from astropy.visualization import ImageNormalize, LogStretch

    flux = _write_image(tmp_path / "broad_flux.img", np.full((4, 5), 2.0))
    product = _flux_product(tmp_path, flux=flux)
    axes = Mock()

    result = product.plot(ax=axes, title="SS 433 soft")

    assert result is axes
    (data,), kwargs = axes.imshow.call_args
    assert data.shape == (4, 5)
    assert kwargs["origin"] == "lower"
    assert kwargs["cmap"] == "inferno"
    assert isinstance(kwargs["norm"], ImageNormalize)
    assert isinstance(kwargs["norm"].stretch, LogStretch)
    axes.figure.colorbar.assert_called_once()
    axes.set_title.assert_called_once_with("SS 433 soft")


def test_flux_image_plot_masks_non_positive_values_for_log(tmp_path: Path) -> None:
    flux = _write_image(
        tmp_path / "broad_flux.img",
        np.array([[0.0, -1.0], [1e-6, 1e-3]]),
    )
    axes = Mock()

    _flux_product(tmp_path, flux=flux).plot(ax=axes)

    (data,), kwargs = axes.imshow.call_args
    assert np.isnan(data).sum() == 2
    assert kwargs["norm"].vmin > 0


def test_flux_image_plot_displays_other_components(tmp_path: Path) -> None:
    from astropy.visualization import SqrtStretch

    counts = _write_image(tmp_path / "broad_thresh.img", np.ones((3, 3)))
    exposure = _write_image(tmp_path / "broad_thresh.expmap", np.ones((3, 3)))
    product = _flux_product(tmp_path, counts=counts, exposure=exposure)

    for component in ("counts", "exposure"):
        axes = Mock()
        product.plot(ax=axes, component=component, stretch="sqrt")
        (_,), kwargs = axes.imshow.call_args
        assert isinstance(kwargs["norm"].stretch, SqrtStretch)


def test_flux_image_plot_requires_an_available_component(tmp_path: Path) -> None:
    flux = _write_image(tmp_path / "broad_flux.img", np.ones((3, 3)))
    product = _flux_product(tmp_path, flux=flux)

    with pytest.raises(ValueError, match="not available"):
        product.plot(ax=Mock(), component="counts")
    with pytest.raises(ValueError, match="unknown image component"):
        product.plot(ax=Mock(), component="rate")
    with pytest.raises(ValueError, match="stretch must be one of"):
        product.plot(ax=Mock(), stretch="power")


def test_flux_image_plot_rejects_empty_images(tmp_path: Path) -> None:
    flux = _write_image(tmp_path / "broad_flux.img", np.zeros((3, 3)))
    product = _flux_product(tmp_path, flux=flux)

    with pytest.raises(ValueError, match="no displayable data"):
        product.plot(ax=Mock())


def test_flux_image_plot_applies_gaussian_smoothing(tmp_path: Path) -> None:
    flux = _write_image(
        tmp_path / "broad_flux.img",
        np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    axes = Mock()

    _flux_product(tmp_path, flux=flux).plot(ax=axes, smooth=1.0)

    (data,), _ = axes.imshow.call_args
    assert data[0, 2] < 10.0  # the peak is spread out
    assert data[1, 2] > 0.0   # flux moved into neighbouring pixels
    assert np.isfinite(data).all()


def test_flux_image_plot_fills_masked_edges_when_smoothing(tmp_path: Path) -> None:
    raw = np.ones((5, 5))
    raw[0, 0] = np.nan
    flux = _write_image(tmp_path / "broad_flux.img", raw)
    axes = Mock()

    _flux_product(tmp_path, flux=flux).plot(ax=axes, smooth=1.0)

    (data,), _ = axes.imshow.call_args
    assert np.isfinite(data[0, 0])  # NaN pixels are interpolated


def test_flux_image_plot_rejects_non_positive_smoothing(tmp_path: Path) -> None:
    flux = _write_image(tmp_path / "broad_flux.img", np.ones((3, 3)))
    product = _flux_product(tmp_path, flux=flux)

    with pytest.raises(ValueError, match="smooth must be positive"):
        product.plot(ax=Mock(), smooth=0.0)
