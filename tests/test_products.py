import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from xray_tools import (
    CIAOGTI,
    CIAOBlankSky,
    CIAOBlankSkyImage,
    CIAOFluxImage,
    CIAOImage,
    CIAOLightCurve,
    CIAOSpectrum,
    CIAOWorkflowError,
    LightCurveCuts,
)
from xray_tools.products.gti import _parse_userlimit_cuts
from xray_tools.products.image import _discover_flux_images
from xray_tools.products.spectrum import _discover_spectrum


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


class FakeImageCrate:
    def __init__(self, data, *, unit=None, keys=None):
        self.data = np.asarray(data, dtype=float)
        self._unit = unit
        self._keys = dict(keys or {})

    def is_image(self):
        return True

    def get_shape(self):
        return self.data.shape

    def get_image(self):
        image = Mock()
        image.get_unit.return_value = self._unit
        return image

    def get_key_value(self, name):
        return self._keys.get(name)


def fake_pycrates(crate):
    pycrates = Mock()
    pycrates.read_file.return_value = crate
    pycrates.get_piximgvals.side_effect = lambda cr: cr.data
    return pycrates


@pytest.fixture
def lightcurve_file(tmp_path: Path) -> Path:
    path = tmp_path / "background.lc"
    path.touch()
    return path


def test_product_paths_are_normalized() -> None:
    assert CIAOLightCurve("background.lc").path == Path("background.lc")
    assert CIAOGTI("background.gti").path == Path("background.gti")
    assert CIAOImage("broad_flux.img").path == Path("broad_flux.img")
    spectrum = CIAOSpectrum("spec.pi", grouped_source="spec_grp.pi")
    assert spectrum.source == Path("spec.pi")
    assert spectrum.grouped_source == Path("spec_grp.pi")
    assert spectrum.arf is None


def test_lightcurve_to_pandas_handles_optional_columns(
    lightcurve_file: Path,
) -> None:
    time = np.array([1.0, 2.0])
    rate = np.array([0.1, 0.2])
    crate = FakeCrate({"TIME": time, "RATE": rate})
    pycrates = Mock()
    pycrates.read_file.return_value = crate

    with patch("xray_tools.products.lightcurve.require_pycrates", return_value=pycrates):
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

    with patch("xray_tools.products.lightcurve.require_lc_clean", return_value=lc_clean):
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
        patch("xray_tools.products.lightcurve.require_lc_clean", return_value=lc_clean),
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

    with patch("xray_tools.products.lightcurve.require_pycrates", return_value=pycrates):
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
    (err_x, _err_y, *_), err_kwargs = axes.errorbar.call_args
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

    with patch("xray_tools.products.gti.require_pycrates", return_value=pycrates):
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
        patch("xray_tools.products.gti.require_pycrates", return_value=pycrates),
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

    with patch("xray_tools.products.gti.require_pycrates", return_value=pycrates):
        cuts = CIAOGTI(gti_file).cuts

    pycrates.read_file.assert_called_once_with(f"{gti_file}[FILTER]")
    assert cuts == LightCurveCuts("count_rate", 0.1, 0.5)


def test_gti_cuts_is_none_without_filter_history(tmp_path: Path) -> None:
    gti_file = tmp_path / "plain.gti"
    gti_file.touch()
    pycrates = Mock()
    pycrates.read_file.side_effect = OSError("no FILTER block")

    with patch("xray_tools.products.gti.require_pycrates", return_value=pycrates):
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


def test_ciao_image_to_numpy_returns_raw_and_masked_arrays(tmp_path: Path) -> None:
    path = tmp_path / "broad_flux.img"
    path.touch()
    crate = FakeImageCrate(np.array([[1.0, np.nan], [0.0, 4.0]]))
    image = CIAOImage(path)

    with patch("xray_tools.products.image.require_pycrates", return_value=fake_pycrates(crate)):
        raw = image.to_numpy()
        masked = image.to_numpy(masked=True)

    assert raw.shape == (2, 2)
    assert np.isnan(raw[0, 1])
    assert np.ma.is_masked(masked[0, 1])
    assert float(masked.mean()) == pytest.approx(5.0 / 3.0)


def test_ciao_image_to_numpy_rejects_non_image_files(tmp_path: Path) -> None:
    path = tmp_path / "events.fits"
    path.touch()
    crate = Mock()
    crate.is_image.return_value = False

    with (
        patch("xray_tools.products.image.require_pycrates", return_value=fake_pycrates(crate)),
        pytest.raises(CIAOWorkflowError, match="not a 2D image"),
    ):
        CIAOImage(path).to_numpy()


def test_ciao_image_exposes_metadata(tmp_path: Path) -> None:
    path = tmp_path / "broad_flux.img"
    path.touch()
    crate = FakeImageCrate(
        np.ones((4, 5)),
        unit="/cm**2 /s",
        keys={"CONTENT": "EVT2", "EXPOSURE": 67045.4, "ENERGYLO": 0.5, "ENERGYHI": 1.5},
    )
    image = CIAOImage(path)

    with patch("xray_tools.products.image.require_pycrates", return_value=fake_pycrates(crate)):
        assert image.shape == (4, 5)
        assert image.unit == "/cm**2 /s"
        assert image.content == "EVT2"
        assert image.exposure == pytest.approx(67045.4)
        assert image.energy_range == (0.5, 1.5)


def test_ciao_image_metadata_defaults_to_none(tmp_path: Path) -> None:
    path = tmp_path / "plain.img"
    path.touch()
    crate = FakeImageCrate(np.ones((2, 2)))
    image = CIAOImage(path)

    with patch("xray_tools.products.image.require_pycrates", return_value=fake_pycrates(crate)):
        assert image.unit is None
        assert image.content is None
        assert image.exposure is None
        assert image.energy_range is None


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

    assert images["hard"].counts == CIAOImage(counts)
    assert images["hard"].exposure == CIAOImage(exposure)
    assert images["hard"].fov is None
    assert images["hard"].psf is None


def test_discover_flux_images_finds_psf_maps(tmp_path: Path) -> None:
    _touch(tmp_path / "ss433_broad_flux.img")
    psf = _touch(tmp_path / "ss433_broad_thresh.psfmap")

    images = _discover_flux_images(tmp_path / "ss433", since=time.time() - 5)

    assert images["broad"].psf == CIAOImage(psf)


def test_discover_flux_images_with_directory_outroot(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()
    flux = _touch(outdir / "0.5-2_flux.img")
    fov = _touch(outdir / "3843.fov")

    images = _discover_flux_images(f"{outdir}/", since=time.time() - 5)

    assert list(images) == ["0.5-2"]
    product = images["0.5-2"]
    assert product.band == "0.5-2"
    assert product.flux == CIAOImage(flux)
    assert product.fov == fov


def _write_image(path: Path, data) -> Path:
    from astropy.io import fits

    fits.PrimaryHDU(np.asarray(data, dtype=float)).writeto(path)
    return path


def patch_image_data(data):
    """Patch CIAOImage.to_numpy so plot tests need no pycrates backend."""
    return patch.object(
        CIAOImage, "to_numpy", return_value=np.asarray(data, dtype=float)
    )


def _flux_product(tmp_path: Path, **files: Path) -> CIAOFluxImage:
    defaults = {"counts": None, "flux": None, "exposure": None, "fov": None, "psf": None}
    defaults.update(files)
    return CIAOFluxImage(band="broad", **defaults)


def test_flux_image_wraps_component_paths(tmp_path: Path) -> None:
    flux = tmp_path / "broad_flux.img"
    flux.touch()
    product = _flux_product(tmp_path, flux=flux)

    assert isinstance(product.flux, CIAOImage)
    assert product.flux.path == flux


def test_flux_image_plot_displays_flux_with_log_stretch(tmp_path: Path) -> None:
    from astropy.visualization import ImageNormalize, LogStretch

    flux = _write_image(tmp_path / "broad_flux.img", np.full((4, 5), 2.0))
    product = _flux_product(tmp_path, flux=flux)
    axes = Mock()

    with patch_image_data(np.full((4, 5), 2.0)):
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
    raw = np.array([[0.0, -1.0], [1e-6, 1e-3]])
    flux = _write_image(tmp_path / "broad_flux.img", raw)
    axes = Mock()

    with patch_image_data(raw):
        _flux_product(tmp_path, flux=flux).plot(ax=axes)

    (data,), kwargs = axes.imshow.call_args
    assert np.isnan(data).sum() == 2
    assert kwargs["norm"].vmin > 0


def test_ciao_image_plot_uses_given_vmin_vmax(tmp_path: Path) -> None:
    flux = _write_image(tmp_path / "broad_flux.img", np.full((3, 3), 2.0))
    axes = Mock()

    with patch_image_data(np.full((3, 3), 2.0)):
        CIAOImage(flux).plot(ax=axes, vmin=0.5, vmax=3.0)

    (_,), kwargs = axes.imshow.call_args
    assert kwargs["norm"].vmin == 0.5
    assert kwargs["norm"].vmax == 3.0


def test_flux_image_plot_displays_other_components(tmp_path: Path) -> None:
    from astropy.visualization import SqrtStretch

    counts = _write_image(tmp_path / "broad_thresh.img", np.ones((3, 3)))
    exposure = _write_image(tmp_path / "broad_thresh.expmap", np.ones((3, 3)))
    product = _flux_product(tmp_path, counts=counts, exposure=exposure)

    for component in ("counts", "exposure"):
        axes = Mock()
        with patch_image_data(np.ones((3, 3))):
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

    with (
        patch_image_data(np.zeros((3, 3))),
        pytest.raises(ValueError, match="no displayable data"),
    ):
        product.plot(ax=Mock())


def test_flux_image_plot_applies_gaussian_smoothing(tmp_path: Path) -> None:
    raw = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    flux = _write_image(tmp_path / "broad_flux.img", raw)
    axes = Mock()

    with patch_image_data(raw):
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

    with patch_image_data(raw):
        _flux_product(tmp_path, flux=flux).plot(ax=axes, smooth=1.0)

    (data,), _ = axes.imshow.call_args
    assert np.isfinite(data[0, 0])  # NaN pixels are interpolated


def test_flux_image_plot_rejects_non_positive_smoothing(tmp_path: Path) -> None:
    flux = _write_image(tmp_path / "broad_flux.img", np.ones((3, 3)))
    product = _flux_product(tmp_path, flux=flux)

    with pytest.raises(ValueError, match="smooth must be positive"):
        product.plot(ax=Mock(), smooth=0.0)


def _key_crate(keys: dict) -> Mock:
    crate = Mock()
    crate.get_keynames.return_value = list(keys)
    crate.get_key_value.side_effect = lambda name: keys.get(name)
    return crate


_BLANK_KEYS = {
    "BKGMETH": "particle",
    "BKGPARAM": "[energy=9000:12000]",
    "BKGSCAL0": 0.0394,
    "BKGSCAL1": 0.0406,
    "BKGSCAL7": 0.1758,
}


def test_blanksky_exposes_scaling_metadata(tmp_path: Path) -> None:
    path = tmp_path / "blank_evt.fits"
    path.touch()
    crate = _key_crate(_BLANK_KEYS)
    blank = CIAOBlankSky(path)

    with patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)):
        method = blank.method
        bkgparams = blank.bkgparams
        scales = blank.scales

    assert method == "particle"
    assert bkgparams == "[energy=9000:12000]"
    assert scales == {0: 0.0394, 1: 0.0406, 7: 0.1758}


def test_blanksky_scales_empty_for_single_chip_files(tmp_path: Path) -> None:
    path = tmp_path / "hrc_blank.fits"
    path.touch()
    crate = _key_crate({"BKGMETH": "particle", "BKGSCALE": 0.5})
    blank = CIAOBlankSky(path)

    with patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)):
        scales = blank.scales

    assert scales == {}


def test_blanksky_method_requires_a_tailored_file(tmp_path: Path) -> None:
    path = tmp_path / "plain_evt.fits"
    path.touch()
    crate = _key_crate({})

    with (
        patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)),
        pytest.raises(ValueError, match="BKGMETH"),
    ):
        _ = CIAOBlankSky(path).method


def test_blanksky_images_run_blanksky_image_and_collect_outputs(
    tmp_path: Path,
) -> None:
    bkg = tmp_path / "blank_evt.fits"
    bkg.touch()
    reference = tmp_path / "0.5-7_thresh.img"
    reference.touch()
    outroot = tmp_path / "obs_bkg"
    crate = _key_crate({"BKGMETH": "particle"})

    def fake_run(**_kwargs):
        for suffix in ("bgnd", "bkgsub"):
            (tmp_path / f"obs_bkg_particle_{suffix}.img").touch()

    runtool = Mock()
    runtool.blanksky_image.side_effect = fake_run

    with (
        patch("xray_tools.products.blanksky.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)),
    ):
        result = CIAOBlankSky(bkg).images(outroot, reference, verbose=2, clobber=True)

    kwargs = runtool.blanksky_image.call_args.kwargs
    assert kwargs["bkgfile"] == str(bkg)
    assert kwargs["outroot"] == str(outroot)
    assert kwargs["imgfile"] == str(reference)
    assert kwargs["verbose"] == 2
    assert kwargs["clobber"] is True
    assert result == CIAOBlankSkyImage(
        background=tmp_path / "obs_bkg_particle_bgnd.img",
        subtracted=tmp_path / "obs_bkg_particle_bkgsub.img",
    )
    assert isinstance(result.background, CIAOImage)
    assert isinstance(result.subtracted, CIAOImage)


def test_blanksky_images_accept_a_ciao_image_reference(tmp_path: Path) -> None:
    bkg = tmp_path / "blank_evt.fits"
    bkg.touch()
    reference = tmp_path / "counts.img"
    reference.touch()
    crate = _key_crate({"BKGMETH": "particle"})
    runtool = Mock()

    def fake_run(**_kwargs):
        for suffix in ("bgnd", "bkgsub"):
            (tmp_path / f"obs_particle_{suffix}.img").touch()

    runtool.blanksky_image.side_effect = fake_run

    with (
        patch("xray_tools.products.blanksky.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)),
    ):
        CIAOBlankSky(bkg).images(tmp_path / "obs", CIAOImage(reference))

    assert runtool.blanksky_image.call_args.kwargs["imgfile"] == str(reference)


def test_blanksky_images_raise_when_outputs_are_missing(tmp_path: Path) -> None:
    bkg = tmp_path / "blank_evt.fits"
    bkg.touch()
    reference = tmp_path / "counts.img"
    reference.touch()
    crate = _key_crate({"BKGMETH": "particle"})
    runtool = Mock()

    with (
        patch("xray_tools.products.blanksky.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)),
        pytest.raises(CIAOWorkflowError, match="no bgnd image"),
    ):
        CIAOBlankSky(bkg).images(tmp_path / "obs", reference)


def test_blanksky_images_refuse_implicit_overwrite(tmp_path: Path) -> None:
    bkg = tmp_path / "blank_evt.fits"
    bkg.touch()
    reference = tmp_path / "counts.img"
    reference.touch()
    (tmp_path / "obs_particle_bgnd.img").touch()
    crate = _key_crate({"BKGMETH": "particle"})
    runtool = Mock()

    with (
        patch("xray_tools.products.blanksky.require_ciao_runtool", return_value=runtool),
        patch("xray_tools.products.blanksky.require_pycrates", return_value=fake_pycrates(crate)),
        pytest.raises(FileExistsError, match="already exists"),
    ):
        CIAOBlankSky(bkg).images(tmp_path / "obs", reference)

    runtool.blanksky_image.assert_not_called()


def _touch_spectrum_products(directory: Path, prefix: str, *names: str) -> dict[str, Path]:
    return {name: _touch(directory / f"{prefix}{name}") for name in names}


_SPECTRUM_NAMES = (
    ".pi",
    ".arf",
    ".rmf",
    "_grp.pi",
    "_bkg.pi",
    "_bkg.arf",
    "_bkg.rmf",
    "_bkg_grp.pi",
)


def test_discover_spectrum_maps_products_and_skips_stale(tmp_path: Path) -> None:
    files = _touch_spectrum_products(tmp_path, "spec", *_SPECTRUM_NAMES)
    _touch(tmp_path / "other.pi", stale=True)
    _touch(tmp_path / "spec.corr.arf", stale=True)

    spectrum = _discover_spectrum(tmp_path / "spec", since=time.time() - 5)

    assert spectrum == CIAOSpectrum(
        source=files[".pi"],
        arf=files[".arf"],
        rmf=files[".rmf"],
        background=files["_bkg.pi"],
        grouped_source=files["_grp.pi"],
        grouped_background=files["_bkg_grp.pi"],
        bkg_arf=files["_bkg.arf"],
        bkg_rmf=files["_bkg.rmf"],
    )


def test_discover_spectrum_prefers_the_corrected_arf(tmp_path: Path) -> None:
    files = _touch_spectrum_products(tmp_path, "spec", ".pi", ".arf", ".corr.arf")

    spectrum = _discover_spectrum(tmp_path / "spec", since=time.time() - 5)

    assert spectrum.arf == files[".corr.arf"]
    assert spectrum.background is None
    assert spectrum.grouped_source is None
    assert spectrum.bkg_arf is None


def test_discover_spectrum_supports_nested_prefixes(tmp_path: Path) -> None:
    directory = tmp_path / "spectrum"
    directory.mkdir()
    files = _touch_spectrum_products(directory, "3843", ".pi", ".rmf")

    spectrum = _discover_spectrum(directory / "3843", since=time.time() - 5)

    assert spectrum.source == files[".pi"]
    assert spectrum.rmf == files[".rmf"]
    assert spectrum.arf is None


def test_discover_spectrum_requires_a_source_spectrum(tmp_path: Path) -> None:
    _touch(tmp_path / "spec_bkg.pi")

    with pytest.raises(CIAOWorkflowError, match="no source spectrum"):
        _discover_spectrum(tmp_path / "spec", since=time.time() - 5)


def test_spectrum_exposes_header_metadata(tmp_path: Path) -> None:
    path = tmp_path / "spec.pi"
    path.touch()
    crate = _key_crate({"EXPOSURE": 67045.4, "BACKSCAL": 2.5e-6, "TOTCTS": 1234})
    spectrum = CIAOSpectrum(path)

    with patch("xray_tools.products.spectrum.require_pycrates", return_value=fake_pycrates(crate)):
        assert spectrum.exposure == pytest.approx(67045.4)
        assert spectrum.backscal == pytest.approx(2.5e-6)
        assert spectrum.counts == 1234


def test_spectrum_metadata_defaults_to_none(tmp_path: Path) -> None:
    path = tmp_path / "spec.pi"
    path.touch()
    crate = _key_crate({})

    with patch("xray_tools.products.spectrum.require_pycrates", return_value=fake_pycrates(crate)):
        spectrum = CIAOSpectrum(path)
        assert spectrum.exposure is None
        assert spectrum.backscal is None
        assert spectrum.counts is None


def test_spectrum_to_sherpa_prefers_the_grouped_spectrum(tmp_path: Path) -> None:
    source = tmp_path / "spec.pi"
    source.touch()
    grouped = tmp_path / "spec_grp.pi"
    grouped.touch()
    sherpa_io = Mock()

    with patch("xray_tools.products.spectrum.require_sherpa", return_value=sherpa_io):
        result = CIAOSpectrum(source, grouped_source=grouped).to_sherpa(use_errors=True)

    sherpa_io.read_pha.assert_called_once_with(str(grouped), use_errors=True)
    assert result is sherpa_io.read_pha.return_value


def test_spectrum_to_sherpa_falls_back_to_the_ungrouped_spectrum(tmp_path: Path) -> None:
    source = tmp_path / "spec.pi"
    source.touch()
    grouped = tmp_path / "spec_grp.pi"
    grouped.touch()
    sherpa_io = Mock()

    with patch("xray_tools.products.spectrum.require_sherpa", return_value=sherpa_io):
        CIAOSpectrum(source, grouped_source=grouped).to_sherpa(prefer_grouped=False)
        CIAOSpectrum(source).to_sherpa()

    first, second = sherpa_io.read_pha.call_args_list
    assert first.args == (str(source),)
    assert second.args == (str(source),)
