from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube
from punchbowl.data import NormalizedMetadata

from simpunch.level2 import add_starfield_clear


@pytest.fixture
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


def test_starfield(sample_ndcube: NDCube) -> None:
    """Test starfield generation."""
    input_data = sample_ndcube((2048,2048))

    output_data = add_starfield_clear(input_data)

    assert isinstance(output_data, NDCube)
