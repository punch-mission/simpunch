"""Creates disortion models."""
import os
from datetime import datetime

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, DistortionLookupTable

from simpunch.level1 import generate_spacecraft_wcs

CURRENT_DIR = os.path.dirname(__file__)

now = datetime(2024, 1, 1, 1, 1, 1)
now_str = now.strftime("%Y%m%d%H%M%S")

for spacecraft_id in ["1", "2", "3", "4"]:

    filename_distortion = (
            os.path.join(CURRENT_DIR, "../simpunch/data/distortion_NFI.fits")
            if spacecraft_id == "4"
            else os.path.join(CURRENT_DIR, "../simpunch/data/distortion_WFI.fits")
        )

    spacecraft_wcs = generate_spacecraft_wcs(spacecraft_id, 0)

    with fits.open(filename_distortion) as hdul:
        err_x = hdul[1].data
        err_y = hdul[2].data

    crpix = err_x.shape[1] / 2 + 0.5, err_x.shape[0] / 2 + 0.5
    crval = 1024.5, 1024.5
    cdelt = (spacecraft_wcs.wcs.cdelt[1] * err_x.shape[1] / 2048,
             spacecraft_wcs.wcs.cdelt[0] * err_x.shape[0] / 2048)

    cpdis1 = DistortionLookupTable(
        -err_x.astype(np.float32), crpix, crval, cdelt,
    )
    cpdis2 = DistortionLookupTable(
        -err_y.astype(np.float32), crpix, crval, cdelt,
    )

    w = WCS(naxis=2)
    w.cpdis1 = cpdis1
    w.cpdis2 = cpdis2
    w.to_fits().writeto(f"PUNCH_DD{spacecraft_id}_{now_str}_v1.fits", overwrite=True)
