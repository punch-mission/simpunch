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

        metaz = NormalizedMetadata.load_template("PZ1", "1")
        metaz["DATE-OBS"] = str(datetime(2024, 2, 22, 16, 0, 1))
        metaz["FILEVRSN"] = "1"
        metaz["POLAROFF"] = 1.0
        metaz["POLARREF"] = "Solar"

        metam = NormalizedMetadata.load_template("PM1", "1")
        metam["DATE-OBS"] = str(datetime(2024, 2, 22, 16, 0, 1))
        metam["FILEVRSN"] = "1"
        metam["POLAROFF"] = 1.0
        metam["POLARREF"] = "Solar"

        metap = NormalizedMetadata.load_template("PP1", "1")
        metap["DATE-OBS"] = str(datetime(2024, 2, 22, 16, 0, 1))
        metap["FILEVRSN"] = "1"
        metap["POLAROFF"] = 1.0
        metap["POLARREF"] = "Solar"

        input_data_z = NDCube(np.ones(shape).astype(np.float32), wcs=wcs, meta=metaz)
        input_data_m = NDCube(np.ones(shape).astype(np.float32), wcs=wcs, meta=metam)
        input_data_p = NDCube(np.ones(shape).astype(np.float32), wcs=wcs, meta=metap)

        return NDCollection(
            [("M", input_data_m),
             ("Z", input_data_z),
             ("P", input_data_p)],
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
    shape = (128, 128)
    input_data = sample_ndcollection(shape)
    output_data = add_starfield_polarized(input_data, polfactor=(1, 1, 1))
    expected_data = np.full((3, 128, 128), 1.0, dtype=np.float32)
    assert isinstance(output_data, NDCollection)
    for k, i in zip(list(output_data.keys()), range(3), strict=False):
        assert np.allclose(output_data[k].data, expected_data[i])
