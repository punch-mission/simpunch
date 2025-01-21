from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata

from simpunch.level0 import add_stray_light
from simpunch.util import generate_stray_light


@pytest.fixture()
def sample_ndcube() -> NDCube:
    def _sample_ndcube(shape: tuple, code:str = "CR1", level:str = "0") -> NDCube:
        data = np.random.random(shape).astype(np.float32) * 1e-12
        sqrt_abs_data = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(np.interp(sqrt_abs_data, (sqrt_abs_data.min(), sqrt_abs_data.max()),
                                                  (0,1)).astype(np.float32))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        meta = NormalizedMetadata.load_template(code, level)
        meta["DATE-OBS"] = str(datetime(2024, 2, 22, 16, 0, 1))
        meta["FILEVRSN"] = "1"
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
    return _sample_ndcube


def test_generate_stray_light() -> None:
    shape = (2048,2048)

    stray_light_wfi = generate_stray_light(shape=shape, instrument="WFI")
    stray_light_nfi = generate_stray_light(shape=shape, instrument="NFI")

    assert np.sum(stray_light_wfi[0]) != 0
    assert np.sum(stray_light_wfi[1]) != 0
    assert np.sum(stray_light_nfi[0]) != 0
    assert np.sum(stray_light_nfi[1]) != 0

    assert stray_light_wfi[0].shape == shape
    assert stray_light_wfi[1].shape == shape
    assert stray_light_nfi[0].shape == shape
    assert stray_light_nfi[1].shape == shape


def test_stray_light(sample_ndcube: NDCube) -> None:
    """Test stray light addition."""
    input_data = sample_ndcube((2048,2048))

    input_data_array = input_data.data.copy()

    stray_light_data = add_stray_light(input_data, polar="clear")

    assert isinstance(stray_light_data, NDCube)
    assert np.sum(stray_light_data.data - input_data_array) != 0
    assert (stray_light_data.data != input_data_array).all()
