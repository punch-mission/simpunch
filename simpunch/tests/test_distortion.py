from datetime import datetime

import astropy.units as u
import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from simpunch.level1 import add_distortion
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata


@pytest.fixture
def sample_ndcube():
    def _sample_ndcube(shape, code="PM1", level="0"):
        data = np.random.random(shape).astype(np.float32)
        sqrt_abs_data = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(np.interp(sqrt_abs_data, (sqrt_abs_data.min(), sqrt_abs_data.max()), (0,1)).astype(np.float32))
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = "HPLN-ARC", "HPLT-ARC"
        wcs.wcs.cunit = "deg", "deg"
        wcs.wcs.cdelt = 0.1, 0.1
        wcs.wcs.crpix = 0, 0
        wcs.wcs.crval = 1, 1
        wcs.wcs.cname = "HPC lon", "HPC lat"

        meta = NormalizedMetadata.load_template(code, level)
        meta['DATE-OBS'] = str(datetime(2024, 2, 22, 16, 0, 1))
        meta['FILEVRSN'] = "1"
        return NDCube(data=data, uncertainty=uncertainty, wcs=wcs, meta=meta)
    return _sample_ndcube


def test_distortion(sample_ndcube) -> None:
    """Test distortion addition."""
    input_data = sample_ndcube((2048,2048))
    distorted_data = add_distortion(input_data)

    assert isinstance(distorted_data, NDCube)
