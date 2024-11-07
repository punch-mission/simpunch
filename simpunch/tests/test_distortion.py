from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata

from simpunch.level1 import add_distortion


@pytest.fixture()
def sample_ndcube() -> NDCube:
    def _sample_ndcube(shape: tuple, code:str = "PM1", level:str = "0") -> NDCube:
        data = np.random.random(shape).astype(np.float32)
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


def test_distortion(sample_ndcube: NDCube) -> None:
    """Test distortion addition."""
    input_data = sample_ndcube((2048,2048))
    original_wcs = input_data.wcs.copy()

    distorted_data = add_distortion(input_data)

    assert isinstance(distorted_data, NDCube)
    assert distorted_data.wcs.has_distortion
    assert (distorted_data.wcs.wcs.cdelt == original_wcs.wcs.cdelt).all()
    assert (distorted_data.wcs.wcs.crpix == original_wcs.wcs.crpix).all()
    assert (distorted_data.wcs.wcs.crval == original_wcs.wcs.crval).all()

    original_coord = original_wcs.pixel_to_world(0, 0)
    distorted_coord = distorted_data.wcs.pixel_to_world(0, 0)

    assert original_coord.Tx.value != distorted_coord.Tx.value
    assert original_coord.Ty.value != distorted_coord.Ty.value
