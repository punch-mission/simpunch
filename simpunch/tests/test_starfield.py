from datetime import datetime

import numpy as np
import pytest
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCollection, NDCube
from punchbowl.data import NormalizedMetadata

from simpunch.level2 import add_starfield_clear, add_starfield_polarized


@pytest.fixture
def sample_ndcube() -> NDCube:
    def _sample_ndcube(shape: tuple, code: str = "PM1", level: str = "0") -> NDCube:
        data = np.random.random(shape).astype(np.float32)
        sqrt_abs_data = np.sqrt(np.abs(data))
        uncertainty = StdDevUncertainty(np.interp(sqrt_abs_data, (sqrt_abs_data.min(), sqrt_abs_data.max()),
                                                  (0, 1)).astype(np.float32))
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


@pytest.fixture
def sample_ndcollection() -> callable:
    def _create_sample_ndcollection(shape: tuple) -> NDCollection:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["HPLN-ARC", "HPLT-ARC"]
        wcs.wcs.cunit = ["deg", "deg"]
        wcs.wcs.cdelt = [0.1, 0.1]
        wcs.wcs.crpix = [shape[1] // 2, shape[0] // 2]
        wcs.wcs.crval = [1, 1]
        wcs.wcs.cname = ["HPC lon", "HPC lat"]

        input_data = NDCube(np.random.random(shape).astype(np.float32), wcs=wcs)

        return NDCollection(
            [("-60.0 deg", input_data),
             ("0.0 deg", input_data),
             ("60.0 deg", input_data)],
            aligned_axes="all",
        )
    return _create_sample_ndcollection



def test_starfield(sample_ndcube: NDCube) -> None:
    """Test starfield generation."""
    input_data = sample_ndcube((2048, 2048))

    output_data = add_starfield_clear(input_data)

    assert isinstance(output_data, NDCube)


def test_polarized_starfield(sample_ndcollection: NDCollection) -> None:
    """Test polarized starfield generation."""
    shape = (2048, 2048)
    input_data = sample_ndcollection(shape)

    output_data = add_starfield_polarized(input_data)

    assert isinstance(output_data, NDCollection)
