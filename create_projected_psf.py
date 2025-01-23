"""Allows for creating a varying PSF reflecting how a true circle looks in an image's projection."""
from pathlib import Path

import numpy as np
import reproject
from astropy.io import fits
from astropy.wcs import WCS
from regularizepsf import (ArrayPSF, simple_functional_psf,
                           varied_functional_psf)
from regularizepsf.util import calculate_covering


def gen_projected_psf_from_image(
        image_path_or_wcs: str | Path | WCS,
        psf_width: int = 64,
        star_gaussian_sigma: float = 3.3/2.355) -> ArrayPSF:
    """Create a varying PSF reflecting how a true circle looks in the image's projection."""
    if isinstance(image_path_or_wcs, WCS):
        wcs = image_path_or_wcs
    else:
        hdul = fits.open(image_path_or_wcs)
        hdr = hdul[1].header
        wcs = WCS(hdr, hdul, key="A")

    # Create a Gaussian star
    coords = np.arange(psf_width) - psf_width / 2 + .5
    xx, yy = np.meshgrid(coords, coords)
    perfect_star = np.exp(-(xx**2 + yy**2) / (2 * star_gaussian_sigma**2))

    star_wcs = WCS(naxis=2)
    star_wcs.wcs.ctype = "RA---ARC", "DEC--ARC"
    star_wcs.wcs.crpix = psf_width / 2 + .5, psf_width / 2 + .5
    star_wcs.wcs.cdelt = wcs.wcs.cdelt

    @simple_functional_psf
    def projected_psf(row: np.ndarray, #noqa: ARG001
                      col: np.ndarray, #noqa: ARG001
                      i: int = 0,
                      j: int = 0) -> np.ndarray:
        # Work out the center of this PSF patch
        ic = i + psf_width / 2 - .5
        jc = j + psf_width / 2 - .5
        ra, dec = wcs.array_index_to_world_values(ic, jc)

        # Create a WCS that places a star at that exact location
        swcs = star_wcs.deepcopy()
        swcs.wcs.crval = ra, dec

        # Project the star into this patch of the full image, telling us what a round
        # star looks like in this projection, distortion, etc.
        psf = reproject.reproject_adaptive(
            (perfect_star, swcs),
            wcs[i:i+psf_width, j:j+psf_width],
            (psf_width, psf_width),
            roundtrip_coords=False, return_footprint=False,
            boundary_mode="grid-constant", boundary_fill_value=0)
        return psf / np.sum(psf)

    @varied_functional_psf(projected_psf)
    def varying_projected_psf(row: int, col: int) -> dict:
        # row and col seem to be the upper-left corner of the image patch we're to describe
        return {"i": row, "j": col}

    coords = calculate_covering(wcs.array_shape, psf_width)
    return varying_projected_psf.as_array_psf(coords, psf_width)
